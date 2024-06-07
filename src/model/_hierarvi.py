import logging
import warnings
from collections.abc import Iterable as IterableClass
from functools import partial
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from mudata import MuData

from scvi import REGISTRY_KEYS
from scvi._types import Number
from scvi._utils import _doc_params
from scvi.data import AnnDataManager, fields
from scvi.data._utils import _check_nonnegative_integers
from scvi.dataloaders import DataSplitter
from scvi.model._utils import (
    _get_batch_code_from_category,
    _get_var_names_from_manager,
    _init_library_size,
    cite_seq_raw_counts_properties,
)
from scvi.model.base._utils import _de_core
from scvi.module import TOTALVAE
from scvi.train import  TrainRunner
from scvi.utils._docstrings import doc_differential_expression, setup_anndata_dsp

from scvi.model import TOTALVI

#from ..module import HierarVAE

from src.module import HierarVAE
from src.train import AdversarialModifiedPlan

logger = logging.getLogger(__name__)



class HierarVI(TOTALVI):
    """Multigrate model.

    :param adata:
        AnnData object that has been registered via :meth:`~multigrate.model.MultiVAE.setup_anndata`.
    :param integrate_on:
        One of the categorical covariates refistered with :math:`~multigrate.model.MultiVAE.setup_anndata` to integrate on. The latent space then will be disentangled from this covariate. If `None`, no integration is performed.
    :param condition_encoders:
        Whether to concatentate covariate embeddings to the first layer of the encoders. Default is `False`.
    :param condition_decoders:
        Whether to concatentate covariate embeddings to the first layer of the decoders. Default is `True`.
    :param normalization:
        What normalization to use; has to be one of `batch` or `layer`. Default is `layer`.
    :param z_dim:
        Dimensionality of the latent space. Default is 15.
    :param losses:
        Which losses to use for each modality. Has to be the same length as the number of modalities. Default is `MSE` for all modalities.
    :param dropout:
        Dropout rate. Default is 0.2.
    :param cond_dim:
        Dimensionality of the covariate embeddings. Default is 10.
    :param loss_coefs:
        Loss coeficients for the different losses in the model. Default is 1 for all.
    :param n_layers_encoders:
        Number of layers for each encoder. Default is 2 for all modalities. Has to be the same length as the number of modalities.
    :param n_layers_decoders:
        Number of layers for each decoder. Default is 2 for all modalities. Has to be the same length as the number of modalities.
    :param n_hidden_encoders:
        Number of nodes for each hidden layer in the encoders. Default is 32.
    :param n_hidden_decoders:
        Number of nodes for each hidden layer in the decoders. Default is 32.
    """

    _module_cls = HierarVAE
    _data_splitter_cls = DataSplitter
    _training_plan_cls = AdversarialModifiedPlan
    _train_runner_cls = TrainRunner

    def __init__(
            self,
            adata: AnnData,
            n_latent: int = 20,
            gene_dispersion: Literal[
                "gene", "gene-batch", "gene-label", "gene-cell"
            ] = "gene",
            protein_dispersion: Literal[
                "protein", "protein-batch", "protein-label"
            ] = "protein",
            gene_likelihood: Literal["zinb", "nb"] = "nb",
            latent_distribution: Literal["normal", "ln"] = "normal",
            empirical_protein_background_prior: Optional[bool] = None,
            override_missing_proteins: bool = False,
            **model_kwargs,
    ):
        super().__init__(adata, n_latent=n_latent, **model_kwargs)

    @torch.inference_mode()
    def get_latent_representation(
            self,
            adata: Optional[AnnData] = None,
            indices: Optional[Sequence[int]] = None,
            give_mean: bool = True,
            mc_samples: int = 5000,
            batch_size: Optional[int] = None,
            return_dist: bool = False,
    ):
        """Return the latent representation for each cell.

        This is typically denoted as :math:`z_n`.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        give_mean
            Give mean of distribution or sample from it.
        mc_samples
            For distributions with no closed-form mean (e.g., `logistic normal`), how many Monte Carlo
            samples to take for computing mean.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_dist
            Return (mean, variance) of distributions instead of just the mean.
            If `True`, ignores `give_mean` and `mc_samples`. In the case of the latter,
            `mc_samples` is used to compute the mean of a transformed distribution.
            If `return_dist` is true the untransformed mean and variance are returned.

        Returns
        -------
        Low-dimensional representation for each cell or a tuple containing its mean and variance.
        """
        self._check_if_trained(warn=False)

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        latent1 = []
        latent2 = []
        latent1r = []
        latent1p = []

        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            if "qz1" in outputs:
                qz1 = outputs["qz1"]
                qz2 = outputs["qz2"]
                qz1r = outputs["qz1r"]
                qz1p = outputs["qz1p"]
                # qzr = outputs["qzr"]
                # qzp = outputs["qzp"]
            else:
                qz_m, qz_v = outputs["qz_m"], outputs["qz_v"]
                qz = torch.distributions.Normal(qz_m, qz_v.sqrt())
            if give_mean:
                # does each model need to have this latent distribution param?
                if self.module.latent_distribution == "ln":
                    samples = qz.sample([mc_samples])
                    z = torch.nn.functional.softmax(samples, dim=-1)
                    z = z.mean(dim=0)
                else:
                    z1 = qz1.loc
                    z2 = qz2.loc
                    z1r = qz1r.loc
                    z1p = qz1p.loc

            else:
                z1 = outputs["z1"]
                z2 = outputs["z2"]
                z1r = outputs["z1r"]
                z1p = outputs["z1p"]

            latent1 += [z1.cpu()]
            latent2 += [z2.cpu()]
            latent1r += [z1r.cpu()]
            latent1p += [z1p.cpu()]

        return (
            torch.cat(latent1).numpy(), torch.cat(latent2).numpy(), torch.cat(latent1r).numpy(),
            torch.cat(latent1p).numpy()
        )

    @torch.inference_mode()
    def posterior_predictive_sample(
            self,
            adata: Optional[AnnData] = None,
            indices: Optional[Sequence[int]] = None,
            n_samples: int = 1,
            batch_size: Optional[int] = None,
            gene_list: Optional[Sequence[str]] = None,
            protein_list: Optional[Sequence[str]] = None,
            swap_latent=False,
    ) -> np.ndarray:
        r"""Generate observation samples from the posterior predictive distribution.

        The posterior predictive distribution is written as :math:`p(\hat{x}, \hat{y} \mid x, y)`.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        n_samples
            Number of required samples for each cell
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        gene_list
            Names of genes of interest
        protein_list
            Names of proteins of interest
        swap_latent
            uses z2 instead of z1 while regenerating gene

        Returns
        -------
        x_new : :class:`~numpy.ndarray`
            tensor with shape (n_cells, n_genes, n_samples)
        """
        if self.module.gene_likelihood not in ["nb"]:
            raise ValueError("Invalid gene_likelihood")

        adata = self._validate_anndata(adata)
        adata_manager = self.get_anndata_manager(adata, required=True)
        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = _get_var_names_from_manager(adata_manager)
            gene_mask = [True if gene in gene_list else False for gene in all_genes]
        if protein_list is None:
            protein_mask = slice(None)
        else:
            all_proteins = self.protein_state_registry.column_names
            protein_mask = [True if p in protein_list else False for p in all_proteins]

        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        scdl_list = []
        for tensors in scdl:
            rna_sample, protein_sample = self.module.sample(
                tensors, n_samples=n_samples, swap_latent=swap_latent
            )
            rna_sample = rna_sample[..., gene_mask]
            protein_sample = protein_sample[..., protein_mask]
            data = torch.cat([rna_sample, protein_sample], dim=-1).numpy()

            scdl_list += [data]
            if n_samples > 1:
                scdl_list[-1] = np.transpose(scdl_list[-1], (1, 2, 0))
        scdl_list = np.concatenate(scdl_list, axis=0)

        return scdl_list
