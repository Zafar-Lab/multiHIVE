from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple, Union

import torch
from scvi import REGISTRY_KEYS
from scvi.distributions import (
    NegativeBinomial,
    ZeroInflatedNegativeBinomial,
    NegativeBinomialMixture,
    Poisson,
)
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from torch import nn
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
from scvi.data._constants import ADATA_MINIFY_TYPE
from scvi.module._constants import MODULE_KEYS

from scvi.module import TOTALVAE

from typing import Dict, Iterable, Literal, Optional, Tuple, Union

from multiHIVE.nn import Encoder, Decoder
import numpy as np
from scvi.nn import one_hot
import torch.nn.functional as F


def _get_dict_if_none(param):
    param = {} if not isinstance(param, dict) else param

    return param


class multiHIVEvae(BaseModuleClass):
    def __init__(
        self,
        n_input_genes: int = 0,
        n_input_regions: int = 0,
        n_input_proteins: int = 0,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 256,
        n_latent: int = 20,
        n_layers_encoder: int = 2,
        n_layers_decoder: int = 2,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate: float = 0.2,
        region_factors: bool = True,
        gene_likelihood: Literal["zinb", "nb"] = "nb",
        gene_dispersion: Literal["gene", "gene-batch", "gene-label"] = "gene",
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        latent_distribution: Literal["normal", "ln"] = "normal",
        deeply_inject_covariates: bool = False,
        encode_covariates: bool = True,  # false not working TODO
        protein_background_prior_mean: Optional[np.ndarray] = None,
        protein_background_prior_scale: Optional[np.ndarray] = None,
        protein_dispersion: Literal["protein", "protein-batch", "protein-label"] = "protein",
        protein_batch_mask: Dict[Union[str, int], np.ndarray] = None,
        log_variational: bool = True,
        library_log_means: Optional[np.ndarray] = None,
        library_log_vars: Optional[np.ndarray] = None,
        kl_dot_product: bool = False,
        deep_network: bool = False,
        use_size_factor_key: bool = False,
        use_observed_lib_size: bool = True,
    ):
        super().__init__()
        # self.n_layers_encoder self.n_layers_decoder self.n_cats_per_cov self.n_continuous_cov
        self.n_input_regions = n_input_regions
        self.n_hidden = n_hidden
        self.region_factors = None

        self.gene_dispersion = gene_dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.n_input_genes = n_input_genes
        self.n_input_proteins = n_input_proteins
        self.protein_dispersion = protein_dispersion
        self.latent_distribution = latent_distribution
        self.protein_batch_mask = protein_batch_mask
        self.encode_covariates = encode_covariates
        self.use_size_factor_key = use_size_factor_key
        self.use_observed_lib_size = use_size_factor_key or use_observed_lib_size

        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_means is None:
                raise ValueError(
                    "If not using observed_lib_size, "
                    "must provide library_log_means and library_log_vars."
                )

            self.register_buffer("library_log_means", torch.from_numpy(library_log_means).float())
            self.register_buffer("library_log_vars", torch.from_numpy(library_log_vars).float())

        # parameters for prior on rate_back (background protein mean)
        if protein_background_prior_mean is None:
            if n_batch > 0:
                self.background_pro_alpha = torch.nn.Parameter(
                    torch.randn(n_input_proteins, n_batch)
                )
                self.background_pro_log_beta = torch.nn.Parameter(
                    torch.clamp(torch.randn(n_input_proteins, n_batch), -10, 1)
                )
            else:
                self.background_pro_alpha = torch.nn.Parameter(torch.randn(n_input_proteins))
                self.background_pro_log_beta = torch.nn.Parameter(
                    torch.clamp(torch.randn(n_input_proteins), -10, 1)
                )
        else:
            if protein_background_prior_mean.shape[1] == 1 and n_batch != 1:
                init_mean = protein_background_prior_mean.ravel()
                init_scale = protein_background_prior_scale.ravel()
            else:
                init_mean = protein_background_prior_mean
                init_scale = protein_background_prior_scale
            self.background_pro_alpha = torch.nn.Parameter(
                torch.from_numpy(init_mean.astype(np.float32))
            )
            self.background_pro_log_beta = torch.nn.Parameter(
                torch.log(torch.from_numpy(init_scale.astype(np.float32)))
            )

        if self.gene_dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes))
        elif self.gene_dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes, n_batch))
        elif self.gene_dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes, n_labels))
        else:  # gene-cell
            pass

        if self.protein_dispersion == "protein":
            self.py_r = torch.nn.Parameter(2 * torch.rand(self.n_input_proteins))
        elif self.protein_dispersion == "protein-batch":
            self.py_r = torch.nn.Parameter(2 * torch.rand(self.n_input_proteins, n_batch))
        elif self.protein_dispersion == "protein-label":
            self.py_r = torch.nn.Parameter(2 * torch.rand(self.n_input_proteins, n_labels))
        else:  # protein-cell
            pass

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        self.deeply_inject_covariates = deeply_inject_covariates
        if region_factors:
            self.region_factors = torch.nn.Parameter(torch.zeros(self.n_input_regions))

        # accessibility
        # accessibility encoder
        if self.n_input_regions == 0:
            input_acc = 0
        else:
            input_acc = self.n_input_regions

        self.n_input_regions = n_input_regions
        n_input = n_input_genes + self.n_input_proteins + input_acc
        n_input_encoder = n_input + n_continuous_cov * encode_covariates

        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        encoder_cat_list = cat_list if encode_covariates else None

        if n_hidden is None:
            if n_input_regions == 0:
                n_hidden = np.min([128, int(np.sqrt(n_input_genes))])
            else:
                n_hidden = np.min([128, int(np.sqrt(self.n_input_regions))])

        self.encoder = Encoder(
            n_input_genes=n_input_genes,
            n_input_proteins=n_input_proteins,
            n_input_regions=input_acc,
            n_input=n_input_encoder,
            n_batch=n_batch,
            n_continuous_cov=n_continuous_cov,
            n_latent=n_latent,
            n_cat_list=encoder_cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            kl_dot_product=kl_dot_product,
            deep_network=deep_network,
            n_cats_per_cov=n_cats_per_cov,
        )
        self.decoder = Decoder(
            n_input=n_latent,
            n_output_genes=n_input_genes,
            n_output_proteins=self.n_input_proteins,
            n_output_regions=self.n_input_regions,
            n_cat_list=cat_list,
            n_layers=n_layers_decoder,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=True,
            use_layer_norm=False,
            scale_activation="softplus" if use_size_factor_key else "softmax",
        )
        print(self.encoder)
        print("decoder")
        print(self.decoder)

    def get_sample_dispersion(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        n_samples: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the tensors of dispersions for genes and proteins.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input_genes)``
        y
            tensor of values with shape ``(batch_size, n_input_proteins)``
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        label
            tensor of cell-types labels with shape ``(batch_size, n_labels)``
        n_samples
            number of samples

        Returns
        -------
        type
            tensors of dispersions of the negative binomial distribution
        """
        outputs = self.inference(x, y, batch_index=batch_index, label=label, n_samples=n_samples)
        px_r = outputs["px_"]["r"]
        py_r = None
        if outputs["py_"]:
            py_r = outputs["py_"]["r"]
        return px_r, py_r

    def get_reconstruction_loss_expression(self, x, px_rate, px_r, px_dropout):
        """Computes the reconstruction loss for the expression data."""
        rl = 0.0
        if self.gene_likelihood == "zinb":
            rl = (
                -ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout)
                .log_prob(x)
                .sum(dim=-1)
            )
        elif self.gene_likelihood == "nb":
            rl = -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x).sum(dim=-1)
        elif self.gene_likelihood == "poisson":
            rl = -Poisson(px_rate).log_prob(x).sum(dim=-1)
        return rl

    def get_reconstruction_loss_accessibility(self, x, p, d):
        """Computes the reconstruction loss for the accessibility data."""
        reg_factor = torch.sigmoid(self.region_factors) if self.region_factors is not None else 1
        return torch.nn.BCELoss(reduction="none")(p * d * reg_factor, (x > 0).float()).sum(dim=-1)

    def get_reconstruction_loss_protein(self, y, py_, pro_batch_mask_minibatch=None):
        """Get the reconstruction loss for protein data."""
        py_conditional = NegativeBinomialMixture(
            mu1=py_["rate_back"],
            mu2=py_["rate_fore"],
            theta1=py_["r"],
            mixture_logits=py_["mixing"],
        )

        reconst_loss_protein_full = -py_conditional.log_prob(y)

        if pro_batch_mask_minibatch is not None:
            temp_pro_loss_full = torch.zeros_like(reconst_loss_protein_full)
            temp_pro_loss_full.masked_scatter_(
                pro_batch_mask_minibatch.bool(), reconst_loss_protein_full
            )
            rl_protein = temp_pro_loss_full.sum(dim=-1)
        else:
            rl_protein = reconst_loss_protein_full.sum(dim=-1)

        return rl_protein

    def _get_inference_input(
        self,
        tensors,
    ) -> dict[str, torch.Tensor | None]:
        """Get input tensors for the inference process."""
        x = tensors.get(REGISTRY_KEYS.X_KEY, None)
        x_rna = x[:, : self.n_input_genes]
        if self.n_input_regions == 0:
            x_atac = torch.zeros(x.shape[0], 1, device=x.device, requires_grad=False)
        else:
            x_atac = x[:, self.n_input_genes : (self.n_input_genes + self.n_input_regions)]
        if self.n_input_proteins == 0:
            y = torch.zeros(x.shape[0], 1, device=x.device, requires_grad=False)
        else:
            y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]

        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        cont_covs = tensors.get(REGISTRY_KEYS.CONT_COVS_KEY)
        cat_covs = tensors.get(REGISTRY_KEYS.CAT_COVS_KEY)
        input_dict = dict(
            x=x_rna,
            y=y,
            c=x_atac,
            batch_index=batch_index,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
        )
        return input_dict

    @auto_move_data
    def inference(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        c: torch.Tensor,  # atac
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
        c_ = c

        library_gene = x.sum(1).unsqueeze(1)

        if self.log_variational:
            x_ = torch.log(1 + x_)
            if self.n_input_proteins > 0:
                y_ = torch.log(1 + y_)
            if self.n_input_regions > 0:
                c_ = torch.log(1 + c_)

        if cont_covs is not None and self.encode_covariates is True:
            if self.n_input_proteins > 0 and self.n_input_regions > 0:
                encoder_input = torch.cat((x_, y_, c_, cont_covs), dim=-1)
            elif self.n_input_proteins > 0:
                encoder_input = torch.cat((x_, y_, cont_covs), dim=-1)
            elif self.n_input_regions > 0:
                encoder_input = torch.cat((x_, c_, cont_covs), dim=-1)
        else:
            if self.n_input_proteins > 0 and self.n_input_regions > 0:
                encoder_input = torch.cat(
                    (
                        x_,
                        y_,
                        c_,
                    ),
                    dim=-1,
                )
            elif self.n_input_proteins > 0:
                encoder_input = torch.cat((x_, y_), dim=-1)
            elif self.n_input_regions > 0:
                encoder_input = torch.cat((x_, c_), dim=-1)
        if cat_covs is not None and self.encode_covariates is True:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()
        qz1, qz2, latent, untran_latent, qz1r, qz1p, qz1a, libsize_acc = self.encoder(
            x_, y_, c_, encoder_input, batch_index, *categorical_input
        )

        z1 = latent["z1"]
        untran_z1 = untran_latent["z1"]

        z2 = latent["z2"]
        untran_z2 = untran_latent["z2"]

        z1r = latent["z1r"]
        untran_z1r = untran_latent["z1r"]

        z1p = latent["z1p"]
        untran_z1p = untran_latent["z1p"]

        z1a = latent["z1a"]
        untran_z1a = untran_latent["z1a"]

        if n_samples > 1:

            def unsqz(zt, n_s):
                return zt.unsqueeze(0).expand((n_s, zt.size(0), zt.size(1)))

            untran_z1 = qz1.sample((n_samples,))
            z1 = self.encoder.z_transformation(untran_z1)

            untran_z2 = qz2.sample((n_samples,))
            z2 = self.encoder.z_transformation(untran_z2)

            untran_z1r = qz1r.sample((n_samples,))
            z1r = self.encoder.zr_transformation(untran_z1r)

            untran_z1p = qz1p.sample((n_samples,))
            z1p = self.encoder.zp_transformation(untran_z1p)

            untran_z1a = qz1a.sample((n_samples,))
            z1a = self.encoder.za_transformation(untran_z1a)

            libsize_acc = unsqz(libsize_acc, n_samples)

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
            "qz1a": qz1a,
            "z1": z1,
            "untran_z1": untran_z1,
            "z2": z2,
            "untran_z2": untran_z2,
            "z1r": z1r,
            "untran_z1r": untran_z1r,
            "z1p": z1p,
            "untran_z1p": untran_z1p,
            "z1a": z1a,
            "untran_z1a": untran_z1a,
            "library_gene": library_gene,
            "libsize_acc": libsize_acc,
            "untran_l": {},
            "kl": latent["kl"],
        }

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z1"]
        z1r = inference_outputs["z1r"]
        z1p = inference_outputs["z1p"]
        z1a = inference_outputs["z1a"]

        library_gene = inference_outputs["library_gene"]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        label = tensors[REGISTRY_KEYS.LABELS_KEY]

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        size_factor_key = REGISTRY_KEYS.SIZE_FACTOR_KEY
        size_factor = tensors[size_factor_key] if size_factor_key in tensors.keys() else None

        return {
            "z": z,
            "zr": z1r,
            "zp": z1p,
            "za": z1a,
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
        zp: Optional[torch.Tensor],
        za: Optional[torch.Tensor],
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
            decoder_input = torch.cat([z, cont_covs.unsqueeze(0).expand(z.size(0), -1, -1)], dim=-1)
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

        px_, py_, pa_, log_pro_back_mean = self.decoder(
            decoder_input, zr, zp, za, size_factor, batch_index, *categorical_input
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
        return dict(
            pa_=pa_,
            px_=px_,
            py_=py_,
            log_pro_back_mean=log_pro_back_mean,
        )

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        pro_recons_weight=1.0,  # double check these defaults
        kl_weight=1.0,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
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
        libsize_acc = inference_outputs["libsize_acc"]

        px_ = generative_outputs["px_"]
        py_ = generative_outputs["py_"]
        pa_ = generative_outputs["pa_"]

        x = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        x_rna = x[:, : self.n_input_genes]

        c = None
        if self.n_input_regions > 0:
            c = x[:, self.n_input_genes : (self.n_input_genes + self.n_input_regions)]
        y = None
        if self.n_input_proteins > 0:
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
        x = x_rna

        if y is not None:
            reconst_loss_protein = self.get_reconstruction_loss_protein(
                y, py_, pro_batch_mask_minibatch
            )
        else:
            reconst_loss_protein = torch.zeros(x.shape[0], device=x.device, requires_grad=False)

        reconst_loss_gene = self.get_reconstruction_loss_expression(
            x, px_["rate"], px_["r"], px_["dropout"]
        )
        if c is not None:
            reconst_loss_accessibility = self.get_reconstruction_loss_accessibility(
                c, pa_["pa"], libsize_acc
            )
        else:
            reconst_loss_accessibility = torch.zeros(
                x.shape[0], device=x.device, requires_grad=False
            )

        # KL Divergence

        kl_div_l_gene = torch.zeros(x.shape[0], device=x.device, requires_grad=False)
        kl_div_back_pro = torch.zeros(x.shape[0], device=x.device, requires_grad=False)

        if y is not None:
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
            + reconst_loss_accessibility
            + (kl_weight) * kl_div_z
            + kl_div_l_gene
            + (kl_weight) * kl_div_back_pro
        )

        reconst_losses = {
            "reconst_loss_gene": reconst_loss_gene,
            "reconst_loss_protein": reconst_loss_protein,
            "reconst_loss_accessibility": reconst_loss_accessibility,
        }
        kl_local = {
            "kl_div_z": kl_div_z,
            "kl_div_l_gene": kl_div_l_gene,
            "kl_div_back_pro": kl_div_back_pro,
        }

        return LossOutput(loss=loss, reconstruction_loss=reconst_losses, kl_local=kl_local)

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
        pa_ = generative_outputs["pa_"]  # TODO

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
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, LossOutput]:
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
    if not ("latent_qzm" in tensors.keys() and "latent_qzv" in tensors.keys()):
        # Remove full_forward_pass if not minified model
        get_inference_input_kwargs.pop("full_forward_pass", None)

    inference_inputs = module._get_inference_input(tensors, **get_inference_input_kwargs)
    inference_outputs = module.inference(**inference_inputs, **inference_kwargs)
    if swap:
        inference_outputs["z1"], inference_outputs["z2"] = (
            inference_outputs["z2"],
            inference_outputs["z1"],
        )

    generative_inputs = module._get_generative_input(
        tensors, inference_outputs, **get_generative_input_kwargs
    )
    generative_outputs = module.generative(**generative_inputs, **generative_kwargs)
    if compute_loss:
        losses = module.loss(tensors, inference_outputs, generative_outputs, **loss_kwargs)
        return inference_outputs, generative_outputs, losses
    else:
        return inference_outputs, generative_outputs
