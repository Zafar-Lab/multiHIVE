from __future__ import annotations

import warnings
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
from scvi import REGISTRY_KEYS
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial, NegativeBinomialMixture
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from torch import nn
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from scvi.module import TOTALVAE

from typing import Dict, Iterable, Literal, Optional, Tuple, Union

from src.nn import Encoder, Decoder
from scvi.autotune._types import Tunable
import numpy as np
from scvi.nn import one_hot
import torch.nn.functional as F

def _get_dict_if_none(param):
    param = {} if not isinstance(param, dict) else param

    return param

class HierarVAE(TOTALVAE):
    def __init__(
            self,
            n_input_genes: int,
            n_input_proteins: int,
            n_batch: int = 0,
            n_labels: int = 0,
            n_hidden: Tunable[int] = 256,
            n_latent: Tunable[int] = 20,
            n_layers_decoder: Tunable[int] = 1,
            n_continuous_cov: int = 0,
            n_cats_per_cov: Optional[Iterable[int]] = None,
            dropout_rate_decoder: Tunable[float] = 0.2,
            dropout_rate_encoder: Tunable[float] = 0.2,
            gene_dispersion: Tunable[Literal["gene", "gene-batch", "gene-label"]] = "gene",
            protein_dispersion: Tunable[
                Literal["protein", "protein-batch", "protein-label"]
            ] = "protein",
            log_variational: bool = True,
            gene_likelihood: Tunable[Literal["zinb", "nb"]] = "nb",
            latent_distribution: Tunable[Literal["normal", "ln"]] = "normal",
            protein_batch_mask: Dict[Union[str, int], np.ndarray] = None,
            encode_covariates: bool = True,
            protein_background_prior_mean: Optional[np.ndarray] = None,
            protein_background_prior_scale: Optional[np.ndarray] = None,
            use_size_factor_key: bool = False,
            use_observed_lib_size: bool = True,
            library_log_means: Optional[np.ndarray] = None,
            library_log_vars: Optional[np.ndarray] = None,
            use_batch_norm: Tunable[Literal["encoder", "decoder", "none", "both"]] = "both",
            use_layer_norm: Tunable[Literal["encoder", "decoder", "none", "both"]] = "none",
            kl_dot_product: bool = False,
            deep_network: bool = False,
    ):
        super().__init__(
            n_input_genes=n_input_genes,
            n_input_proteins=n_input_proteins,
            n_batch=n_batch,
            n_latent=n_latent,
            n_continuous_cov=n_continuous_cov,
            n_cats_per_cov=n_cats_per_cov,
            gene_dispersion=gene_dispersion,
            protein_dispersion=protein_dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            protein_batch_mask=protein_batch_mask,
            protein_background_prior_mean=protein_background_prior_mean,
            protein_background_prior_scale=protein_background_prior_scale,
            use_size_factor_key=use_size_factor_key,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars
        )

        n_input = n_input_genes + self.n_input_proteins
        n_input_encoder = n_input + n_continuous_cov * encode_covariates

        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        encoder_cat_list = cat_list if encode_covariates else None

        self.encoder = Encoder(
            n_input_genes=n_input_genes,
            n_input_proteins=n_input_proteins,
            n_input=n_input_encoder,
            n_latent=n_latent,
            n_cat_list=encoder_cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            distribution=latent_distribution,
            kl_dot_product= kl_dot_product,
            deep_network= deep_network,
        )
        self.decoder = Decoder(
            n_latent + n_continuous_cov,
            n_input_genes,
            self.n_input_proteins,
            n_layers=n_layers_decoder,
            n_cat_list=cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
            use_batch_norm=True,
            use_layer_norm=False,
            scale_activation="softplus" if use_size_factor_key else "softmax",
        )

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z1"]
        z1r = inference_outputs["z1r"]
        z1p = inference_outputs["z1p"]

        library_gene = inference_outputs["library_gene"]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        label = tensors[REGISTRY_KEYS.LABELS_KEY]

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        size_factor_key = REGISTRY_KEYS.SIZE_FACTOR_KEY
        size_factor = (
            tensors[size_factor_key] if size_factor_key in tensors.keys() else None
        )

        return {
            "z": z,
            "zr": z1r,
            "zp": z1p,
            "library_gene": library_gene,
            "batch_index": batch_index,
            "label": label,
            "cat_covs": cat_covs,
            "cont_covs": cont_covs,
            "size_factor": size_factor,

        }

    @auto_move_data
    def generative(
            self,
            z: torch.Tensor,
            zr: torch.Tensor,
            zp: torch.Tensor,
            library_gene: torch.Tensor,
            batch_index: torch.Tensor,
            label: torch.Tensor,
            cont_covs=None,
            cat_covs=None,
            size_factor=None,
            transform_batch: Optional[int] = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Run the generative step."""
        if cont_covs is None:
            decoder_input = z
        elif z.dim() != cont_covs.dim():
            decoder_input = torch.cat(
                [z, cont_covs.unsqueeze(0).expand(z.size(0), -1, -1)], dim=-1
            )
        else:
            decoder_input = torch.cat([z, cont_covs], dim=-1)

        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        if not self.use_size_factor_key:
            size_factor = library_gene

        px_, py_, log_pro_back_mean = self.decoder(
            decoder_input, zr, zp, size_factor, batch_index, *categorical_input
        )

        if self.gene_dispersion == "gene-label":
            # px_r gets transposed - last dimension is nb genes
            px_r = F.linear(one_hot(label, self.n_labels), self.px_r)
        elif self.gene_dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.gene_dispersion == "gene":
            px_r = self.px_r
        px_r = torch.exp(px_r)

        if self.protein_dispersion == "protein-label":
            # py_r gets transposed - last dimension is n_proteins
            py_r = F.linear(one_hot(label, self.n_labels), self.py_r)
        elif self.protein_dispersion == "protein-batch":
            py_r = F.linear(one_hot(batch_index, self.n_batch), self.py_r)
        elif self.protein_dispersion == "protein":
            py_r = self.py_r
        py_r = torch.exp(py_r)

        px_["r"] = px_r
        py_["r"] = py_r
        return {
            "px_": px_,
            "py_": py_,
            "log_pro_back_mean": log_pro_back_mean,
        }

    @auto_move_data
    def inference(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            batch_index: Optional[torch.Tensor] = None,
            label: Optional[torch.Tensor] = None,
            n_samples=1,
            cont_covs=None,
            cat_covs=None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Internal helper function to compute necessary inference quantities.

        We use the dictionary ``px_`` to contain the parameters of the ZINB/NB for genes.
        The rate refers to the mean of the NB, dropout refers to Bernoulli mixing parameters.
        `scale` refers to the quanity upon which differential expression is performed. For genes,
        this can be viewed as the mean of the underlying gamma distribution.

        We use the dictionary ``py_`` to contain the parameters of the Mixture NB distribution for proteins.
        `rate_fore` refers to foreground mean, while `rate_back` refers to background mean. ``scale`` refers to
        foreground mean adjusted for background probability and scaled to reside in simplex.
        ``back_alpha`` and ``back_beta`` are the posterior parameters for ``rate_back``.  ``fore_scale`` is the scaling
        factor that enforces `rate_fore` > `rate_back`.

        ``px_["r"]`` and ``py_["r"]`` are the inverse dispersion parameters for genes and protein, respectively.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input_genes)``
        y
            tensor of values with shape ``(batch_size, n_input_proteins)``
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        label
            tensor of cell-types labels with shape (batch_size, n_labels)
        n_samples
            Number of samples to sample from approximate posterior
        cont_covs
            Continuous covariates to condition on
        cat_covs
            Categorical covariates to condition on
        """
        x_ = x
        y_ = y

        library_gene = x.sum(1).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)
            y_ = torch.log(1 + y_)

        if cont_covs is not None and self.encode_covariates is True:
            encoder_input = torch.cat((x_, y_, cont_covs), dim=-1)
        else:
            encoder_input = torch.cat((x_, y_), dim=-1)
        if cat_covs is not None and self.encode_covariates is True:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()
        qz1, qz2, latent, untran_latent, qz1r, qz1p = self.encoder(
            x_, y_, encoder_input, batch_index, *categorical_input
        )

        z1 = latent["z1"]
        untran_z1 = untran_latent["z1"]

        z2 = latent["z2"]
        untran_z2 = untran_latent["z2"]

        z1r = latent["z1r"]
        untran_z1r = untran_latent["z1r"]

        z1p = latent["z1p"]
        untran_z1p = untran_latent["z1p"]


        if n_samples > 1:
            untran_z1 = qz1.sample((n_samples,))
            z1 = self.encoder.z_transformation(untran_z1)

            untran_z2 = qz2.sample((n_samples,))
            z2 = self.encoder.z_transformation(untran_z2)

            untran_z1r = qz1r.sample((n_samples,))
            z1r = self.encoder.zr_transformation(untran_z1r)

            untran_z1p = qz1p.sample((n_samples,))
            z1p = self.encoder.zp_transformation(untran_z1p)



        # Background regularization
        if self.gene_dispersion == "gene-label":
            # px_r gets transposed - last dimension is nb genes
            px_r = F.linear(one_hot(label, self.n_labels), self.px_r)
        elif self.gene_dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.gene_dispersion == "gene":
            px_r = self.px_r
        px_r = torch.exp(px_r)

        if self.protein_dispersion == "protein-label":
            # py_r gets transposed - last dimension is n_proteins
            py_r = F.linear(one_hot(label, self.n_labels), self.py_r)
        elif self.protein_dispersion == "protein-batch":
            py_r = F.linear(one_hot(batch_index, self.n_batch), self.py_r)
        elif self.protein_dispersion == "protein":
            py_r = self.py_r
        py_r = torch.exp(py_r)
        if self.n_batch > 0:
            py_back_alpha_prior = F.linear(
                one_hot(batch_index, self.n_batch), self.background_pro_alpha
            )
            py_back_beta_prior = F.linear(
                one_hot(batch_index, self.n_batch),
                torch.exp(self.background_pro_log_beta),
            )
        else:
            py_back_alpha_prior = self.background_pro_alpha
            py_back_beta_prior = torch.exp(self.background_pro_log_beta)
        self.back_mean_prior = Normal(py_back_alpha_prior, py_back_beta_prior)

        return {
            "qz1": qz1,
            "qz2": qz2,
            "qz1r": qz1r,
            "qz1p": qz1p,
            "z1": z1,
            "untran_z1": untran_z1,
            "z2": z2,
            "untran_z2": untran_z2,
            "z1r": z1r,
            "untran_z1r": untran_z1r,
            "z1p": z1p,
            "untran_z1p": untran_z1p,
            "library_gene": library_gene,
            "untran_l": {},
            "kl": latent["kl"],

        }

    def loss(
            self,
            tensors,
            inference_outputs,
            generative_outputs,
            pro_recons_weight=1.0,  # double check these defaults
            kl_weight=1.0,
    ) -> Tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor
    ]:
        """Returns the reconstruction loss and the Kullback divergences.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input_genes)``
        y
            tensor of values with shape ``(batch_size, n_input_proteins)``
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        label
            tensor of cell-types labels with shape (batch_size, n_labels)

        Returns
        -------
        type
            the reconstruction loss and the Kullback divergences
        """

        kl_div_z = inference_outputs["kl"]

        px_ = generative_outputs["px_"]
        py_ = generative_outputs["py_"]

        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]

        if self.protein_batch_mask is not None:
            pro_batch_mask_minibatch = torch.zeros_like(y)
            for b in torch.unique(batch_index):
                b_indices = (batch_index == b).reshape(-1)
                pro_batch_mask_minibatch[b_indices] = torch.tensor(
                    self.protein_batch_mask[str(int(b.item()))].astype(np.float32),
                    device=y.device,
                )
        else:
            pro_batch_mask_minibatch = None

        reconst_loss_gene, reconst_loss_protein = self.get_reconstruction_loss(
            x, y, px_, py_, pro_batch_mask_minibatch
        )

        # KL Divergence

        kl_div_l_gene = 0.0

        kl_div_back_pro_full = kl(
            Normal(py_["back_alpha"], py_["back_beta"]), self.back_mean_prior
        )

        if pro_batch_mask_minibatch is not None:
            # kl_div_back_pro = torch.zeros_like(kl_div_back_pro_full)
            # kl_div_back_pro.masked_scatter_(
            #     pro_batch_mask_minibatch.bool(), kl_div_back_pro_full
            # )
            kl_div_back_pro = pro_batch_mask_minibatch.bool() * kl_div_back_pro_full
            kl_div_back_pro = kl_div_back_pro.sum(dim=1)
        else:
            kl_div_back_pro = kl_div_back_pro_full.sum(dim=1)

        loss = torch.mean(
            reconst_loss_gene
            + pro_recons_weight * reconst_loss_protein
            + (kl_weight) * kl_div_z
            + kl_div_l_gene
            + (kl_weight) * kl_div_back_pro
        )

        reconst_losses = {
            "reconst_loss_gene": reconst_loss_gene,
            "reconst_loss_protein": reconst_loss_protein,
        }
        kl_local = {
            "kl_div_z": kl_div_z,
            "kl_div_l_gene": kl_div_l_gene,
            "kl_div_back_pro": kl_div_back_pro,
        }

        return LossOutput(
            loss=loss, reconstruction_loss=reconst_losses, kl_local=kl_local
        )

    @torch.inference_mode()
    def sample(self, tensors, n_samples=1, swap_latent=False):
        """Sample from the generative model."""
        inference_kwargs = {"n_samples": n_samples}
        with torch.inference_mode():
            (
                inference_outputs,
                generative_outputs,
            ) = self.forward(
                tensors,
                inference_kwargs=inference_kwargs,
                compute_loss=False,
                swap=swap_latent,
            )

        px_ = generative_outputs["px_"]
        py_ = generative_outputs["py_"]

        rna_dist = NegativeBinomial(mu=px_["rate"], theta=px_["r"])
        protein_dist = NegativeBinomialMixture(
            mu1=py_["rate_back"],
            mu2=py_["rate_fore"],
            theta1=py_["r"],
            mixture_logits=py_["mixing"],
        )
        rna_sample = rna_dist.sample().cpu()
        protein_sample = protein_dist.sample().cpu()

        return rna_sample, protein_sample

    @auto_move_data
    def forward(
            self,
            tensors,
            get_inference_input_kwargs: dict | None = None,
            get_generative_input_kwargs: dict | None = None,
            inference_kwargs: dict | None = None,
            generative_kwargs: dict | None = None,
            loss_kwargs: dict | None = None,
            compute_loss=True,
            swap=False,
    ) -> (
            tuple[torch.Tensor, torch.Tensor]
            | tuple[torch.Tensor, torch.Tensor, LossOutput]
    ):
        """Forward pass through the network.

        Parameters
        ----------
        tensors
            tensors to pass through
        get_inference_input_kwargs
            Keyword args for ``_get_inference_input()``
        get_generative_input_kwargs
            Keyword args for ``_get_generative_input()``
        inference_kwargs
            Keyword args for ``inference()``
        generative_kwargs
            Keyword args for ``generative()``
        loss_kwargs
            Keyword args for ``loss()``
        compute_loss
            Whether to compute loss on forward pass. This adds
            another return value.
        """
        return _generic_forward(
            self,
            tensors,
            inference_kwargs,
            generative_kwargs,
            loss_kwargs,
            get_inference_input_kwargs,
            get_generative_input_kwargs,
            compute_loss,
            swap,
        )

def _generic_forward(
        module,
        tensors,
        inference_kwargs,
        generative_kwargs,
        loss_kwargs,
        get_inference_input_kwargs,
        get_generative_input_kwargs,
        compute_loss,
        swap=False,
):
    """Core of the forward call shared by PyTorch- and Jax-based modules."""
    inference_kwargs = _get_dict_if_none(inference_kwargs)
    generative_kwargs = _get_dict_if_none(generative_kwargs)
    loss_kwargs = _get_dict_if_none(loss_kwargs)
    get_inference_input_kwargs = _get_dict_if_none(get_inference_input_kwargs)
    get_generative_input_kwargs = _get_dict_if_none(get_generative_input_kwargs)

    inference_inputs = module._get_inference_input(
        tensors, **get_inference_input_kwargs
    )
    inference_outputs = module.inference(**inference_inputs, **inference_kwargs)
    if swap:
        inference_outputs['z1'], inference_outputs['z2'] = inference_outputs['z2'], inference_outputs['z1']

    generative_inputs = module._get_generative_input(
        tensors, inference_outputs, **get_generative_input_kwargs
    )
    generative_outputs = module.generative(**generative_inputs, **generative_kwargs)
    if compute_loss:
        losses = module.loss(
            tensors, inference_outputs, generative_outputs, **loss_kwargs
        )
        return inference_outputs, generative_outputs, losses
    else:
        return inference_outputs, generative_outputs
