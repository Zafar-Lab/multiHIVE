from __future__ import annotations
from multiHIVE.train import AdversarialModifiedPlan
from multiHIVE.module import multiHIVEvae
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
    ProteinObsmField,
)
from scipy.sparse import csr_matrix, vstack

import logging
import warnings
from collections.abc import Iterable as IterableClass
from functools import partial
from typing import TYPE_CHECKING, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from anndata import AnnData

from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager, fields
from scvi.data._constants import ADATA_MINIFY_TYPE
from scvi.data._utils import _check_nonnegative_integers
from scvi.dataloaders import DataSplitter
from scvi.model._utils import (
    _get_batch_code_from_category,
    _get_var_names_from_manager,
    _init_library_size,
    cite_seq_raw_counts_properties,
    get_max_epochs_heuristic,
    use_distributed_sampler,
)
from scvi.model.base._de_core import _de_core
from scvi.train import TrainRunner
from scvi.utils._docstrings import de_dsp, devices_dsp, setup_anndata_dsp

from scvi.model.base import (
    ArchesMixin,
    BaseModelClass,
    UnsupervisedTrainingMixin,
    VAEMixin,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import Literal

    from mudata import MuData

    from scvi._types import AnnOrMuData, Number

logger = logging.getLogger(__name__)

#####

# from ..module import HierarVAE


logger = logging.getLogger(__name__)


class multiHIVE(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass, ArchesMixin):
    """
    multiHIVE model.
    """

    _module_cls = multiHIVEvae
    _data_splitter_cls = DataSplitter
    _training_plan_cls = AdversarialModifiedPlan
    _train_runner_cls = TrainRunner

    def __init__(
        self,
        adata: AnnData,
        n_genes: int,
        n_regions: int,
        n_proteins: int,
        n_latent: int = 20,
        n_hidden: Optional[int] = None,
        n_layers_encoder: int = 2,
        n_layers_decoder: int = 2,
        dropout_rate: float = 0.1,
        region_factors: bool = True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        gene_dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        protein_dispersion: Literal["protein", "protein-batch", "protein-label"] = "protein",
        gene_likelihood: Literal["zinb", "nb"] = "nb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        empirical_protein_background_prior: Optional[bool] = None,
        override_missing_proteins: bool = False,
        deeply_inject_covariates: bool = False,
        encode_covariates: bool = True,  # False error TODO
        fully_paired: bool = False,
        kl_dot_product: bool = False,
        **model_kwargs,
    ):
        super().__init__(adata)
        self.n_genes = n_genes
        self.n_regions = n_regions
        self.n_proteins = n_proteins

        if n_genes is None or n_regions is None:
            assert isinstance(
                adata, MuData
            ), "n_genes and n_regions must be provided if using AnnData"
            n_genes = self.summary_stats.get("n_vars", 0)
            n_regions = self.summary_stats.get("n_atac", 0)

        prior_mean, prior_scale = None, None
        n_cats_per_cov = (
            self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else []
        )

        use_size_factor_key = REGISTRY_KEYS.SIZE_FACTOR_KEY in self.adata_manager.data_registry
        if self.n_proteins > 0:
            self.protein_state_registry = self.adata_manager.get_state_registry(
                REGISTRY_KEYS.PROTEIN_EXP_KEY
            )
            if (
                fields.ProteinObsmField.PROTEIN_BATCH_MASK in self.protein_state_registry
                and not override_missing_proteins
            ):
                batch_mask = self.protein_state_registry.protein_batch_mask
                msg = (
                    "Some proteins have all 0 counts in some batches. "
                    + "These proteins will be treated as missing measurements; however, "
                    + "this can occur due to experimental design/biology. "
                    + "Reinitialize the model with `override_missing_proteins=True`,"
                    + "to override this behavior."
                )
                warnings.warn(msg, UserWarning)
                self._use_adversarial_classifier = True
            else:
                batch_mask = None
                self._use_adversarial_classifier = False

            emp_prior = (
                empirical_protein_background_prior
                if empirical_protein_background_prior is not None
                else (self.summary_stats.n_proteins > 10)
            )
            if emp_prior:
                prior_mean, prior_scale = self._get_totalvi_protein_priors(adata)
            else:
                prior_mean, prior_scale = None, None
        else:
            batch_mask = None
            self._use_adversarial_classifier = False
            prior_mean, prior_scale = None, None

        n_batch = self.summary_stats.n_batch
        use_size_factor_key = REGISTRY_KEYS.SIZE_FACTOR_KEY in self.adata_manager.data_registry
        library_log_means, library_log_vars = None, None
        if not use_size_factor_key:
            library_log_means, library_log_vars = _init_library_size(self.adata_manager, n_batch)

        self.module = self._module_cls(
            n_input_genes=n_genes,
            n_input_regions=n_regions,
            n_input_proteins=n_proteins,
            n_batch=self.summary_stats.n_batch,
            # n_obs=adata.n_obs,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers_encoder=n_layers_encoder,
            n_layers_decoder=n_layers_decoder,
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_cats_per_cov=n_cats_per_cov,
            dropout_rate=dropout_rate,
            region_factors=region_factors,
            gene_likelihood=gene_likelihood,
            gene_dispersion=gene_dispersion,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            use_size_factor_key=use_size_factor_key,
            latent_distribution=latent_distribution,
            deeply_inject_covariates=deeply_inject_covariates,
            encode_covariates=encode_covariates,
            protein_background_prior_mean=prior_mean,
            protein_background_prior_scale=prior_scale,
            protein_dispersion=protein_dispersion,
            protein_batch_mask=batch_mask,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            kl_dot_product=kl_dot_product,
            **model_kwargs,
        )
        self._model_summary_string = (
            "MultiVI Model with INPUTS: n_genes:{}, n_regions:{}, n_proteins:{}\n"
            "n_hidden: {}, n_latent: {}, n_layers_encoder: {}, "
            "n_layers_decoder: {} , dropout_rate: {}, latent_distribution: {}, deep injection: {}, "
            "gene_likelihood: {}, gene_dispersion:{}, protein_dispersion:{}"
        ).format(
            n_genes,
            n_regions,
            n_proteins,
            self.module.n_hidden,
            self.module.n_latent,
            n_layers_encoder,
            n_layers_decoder,
            dropout_rate,
            latent_distribution,
            deeply_inject_covariates,
            gene_likelihood,
            gene_dispersion,
            protein_dispersion,
        )
        self.fully_paired = fully_paired
        self.n_latent = n_latent
        self.init_params_ = self._get_init_params(locals())
        self.n_genes = n_genes
        self.n_regions = n_regions
        self.n_proteins = n_proteins

    @devices_dsp.dedent
    def train(
        self,
        max_epochs: int | None = None,
        lr: float = 4e-3,
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        train_size: float | None = None,
        validation_size: float | None = None,
        shuffle_set_split: bool = True,
        batch_size: int = 256,
        early_stopping: bool = True,
        check_val_every_n_epoch: int | None = None,
        reduce_lr_on_plateau: bool = True,
        n_steps_kl_warmup: int | None = None,
        n_epochs_kl_warmup: int | None = None,
        adversarial_classifier: bool | None = None,
        datasplitter_kwargs: dict | None = None,
        plan_kwargs: dict | None = None,
        external_indexing: list[np.array] = None,
        **kwargs,
    ):
        """Trains the model using amortized variational inference.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset.
        lr
            Learning rate for optimization.
        %(param_accelerator)s
        %(param_devices)s
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        shuffle_set_split
            Whether to shuffle indices before splitting. If `False`, the val, train, and test set
            are split in the sequential order of the data according to `validation_size` and
            `train_size` percentages.
        batch_size
            Minibatch size to use during training.
        early_stopping
            Whether to perform early stopping with respect to the validation set.
        check_val_every_n_epoch
            Check val every n train epochs. By default, val is not checked, unless `early_stopping`
            is `True` or `reduce_lr_on_plateau` is `True`. If either of the latter conditions are
            met, val is checked every epoch.
        reduce_lr_on_plateau
            Reduce learning rate on plateau of validation metric (default is ELBO).
        n_steps_kl_warmup
            Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
            Only activated when `n_epochs_kl_warmup` is set to None. If `None`, defaults
            to `floor(0.75 * adata.n_obs)`.
        n_epochs_kl_warmup
            Number of epochs to scale weight on KL divergences from 0 to 1.
            Overrides `n_steps_kl_warmup` when both are not `None`.
        adversarial_classifier
            Whether to use adversarial classifier in the latent space. This helps mixing when
            there are missing proteins in any of the batches. Defaults to `True` is missing
            proteins are detected.
        datasplitter_kwargs
            Additional keyword arguments passed into :class:`~scvi.dataloaders.DataSplitter`.
        plan_kwargs
            Keyword args for :class:`~scvi.train.AdversarialTrainingPlan`. Keyword arguments passed
            to `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        external_indexing
            A list of data split indices in the order of training, validation, and test sets.
            Validation and test set are not required and can be left empty.
        **kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """

        if adversarial_classifier is None:
            adversarial_classifier = self._use_adversarial_classifier
        n_steps_kl_warmup = (
            n_steps_kl_warmup if n_steps_kl_warmup is not None else int(0.75 * self.adata.n_obs)
        )
        if reduce_lr_on_plateau:
            check_val_every_n_epoch = 1

        update_dict = {
            "lr": lr,
            "adversarial_classifier": adversarial_classifier,
            "reduce_lr_on_plateau": reduce_lr_on_plateau,
            "n_epochs_kl_warmup": n_epochs_kl_warmup,
            "n_steps_kl_warmup": n_steps_kl_warmup,
        }
        if plan_kwargs is not None:
            plan_kwargs.update(update_dict)
        else:
            plan_kwargs = update_dict

        if max_epochs is None:
            max_epochs = get_max_epochs_heuristic(self.adata.n_obs)

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else {}
        datasplitter_kwargs = datasplitter_kwargs or {}

        data_splitter = self._data_splitter_cls(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            shuffle_set_split=shuffle_set_split,
            batch_size=batch_size,
            distributed_sampler=use_distributed_sampler(kwargs.get("strategy", None)),
            external_indexing=external_indexing,
            **datasplitter_kwargs,
        )
        training_plan = self._training_plan_cls(
            self.module,
            n_genes=self.n_genes,
            n_proteins=self.n_proteins,
            n_regions=self.n_regions,
            **plan_kwargs,
        )
        runner = self._train_runner_cls(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            early_stopping=early_stopping,
            check_val_every_n_epoch=check_val_every_n_epoch,
            **kwargs,
        )
        return runner()

    @torch.inference_mode()
    def get_latent_library_size(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        give_mean: bool = True,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Returns the latent library size for each cell.

        This is denoted as :math:`\ell_n` in the totalVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        give_mean
            Return the mean or a sample from the posterior distribution.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        """
        self._check_if_trained(warn=False)

        adata = self._validate_anndata(adata)
        post = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        libraries = []
        for tensors in post:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            if give_mean:
                ql = outputs["ql"]
                library = torch.exp(ql.loc + 0.5 * (ql.scale**2))
            else:
                library = outputs["library_gene"]
            libraries += [library.cpu()]
        return torch.cat(libraries).numpy()

    @torch.inference_mode()
    def get_normalized_expression(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        n_samples_overall: Optional[int] = None,
        transform_batch: Optional[Sequence[Union[Number, str]]] = None,
        gene_list: Optional[Sequence[str]] = None,
        use_z_mean: bool = True,
        n_samples: int = 1,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: bool = False,
    ) -> Union[np.ndarray, pd.DataFrame]:
        r"""
        Returns the normalized (decoded) gene expression.

        This is denoted as :math:`\rho_n` in the scVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch
            Batch to condition on.
            If transform_batch is:

            - None, then real observed batch is used.
            - int, then batch transform_batch is used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        library_size
            Scale the expression frequencies to a common library size.
            This allows gene expression levels to be interpreted on a common scale of relevant
            magnitude. If set to `"latent"`, use the latent library size.
        use_z_mean
            If True, use the mean of the latent distribution, otherwise sample from it
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        self._check_adata_modality_weights(adata)
        adata = self._validate_anndata(adata)
        adata_manager = self.get_anndata_manager(adata, required=True)
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        transform_batch = _get_batch_code_from_category(adata_manager, transform_batch)

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names[: self.n_genes]
            gene_mask = [gene in gene_list for gene in all_genes]

        exprs = []
        for tensors in scdl:
            per_batch_exprs = []
            for batch in transform_batch:
                if batch is not None:
                    batch_indices = tensors[REGISTRY_KEYS.BATCH_KEY]
                    tensors[REGISTRY_KEYS.BATCH_KEY] = torch.ones_like(batch_indices) * batch
                _, generative_outputs = self.module.forward(
                    tensors=tensors,
                    inference_kwargs=dict(n_samples=n_samples),
                    generative_kwargs=dict(use_z_mean=use_z_mean),
                    compute_loss=False,
                )
                output = generative_outputs["px_scale"]
                output = output[..., gene_mask]
                output = output.cpu().numpy()
                per_batch_exprs.append(output)
            per_batch_exprs = np.stack(
                per_batch_exprs
            )  # shape is (len(transform_batch) x batch_size x n_var)
            exprs += [per_batch_exprs.mean(0)]

        if n_samples > 1:
            # The -2 axis correspond to cells.
            exprs = np.concatenate(exprs, axis=-2)
        else:
            exprs = np.concatenate(exprs, axis=0)
        if n_samples > 1 and return_mean:
            exprs = exprs.mean(0)

        if return_numpy:
            return exprs
        else:
            return pd.DataFrame(
                exprs,
                columns=adata.var_names[: self.n_genes][gene_mask],
                index=adata.obs_names[indices],
            )

    def get_protein_foreground_probability(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        transform_batch: Optional[Sequence[Union[Number, str]]] = None,
        protein_list: Optional[Sequence[str]] = None,
        n_samples: int = 1,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
    ):
        r"""
        Returns the foreground probability for proteins.

        This is denoted as :math:`(1 - \pi_{nt})` in the totalVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch
            Batch to condition on.
            If transform_batch is:

            - None, then real observed batch is used
            - int, then batch transform_batch is used
            - List[int], then average over batches in list
        protein_list
            Return protein expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        - **foreground_probability** - probability foreground for each protein

        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        adata = self._validate_anndata(adata)
        post = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        if protein_list is None:
            protein_mask = slice(None)
        else:
            all_proteins = self.protein_state_registry.column_names
            protein_mask = [True if p in protein_list else False for p in all_proteins]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        py_mixings = []
        if not isinstance(transform_batch, IterableClass):
            transform_batch = [transform_batch]

        transform_batch = _get_batch_code_from_category(self.adata_manager, transform_batch)
        for tensors in post:
            y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]
            py_mixing = torch.zeros_like(y[..., protein_mask])
            if n_samples > 1:
                py_mixing = torch.stack(n_samples * [py_mixing])
            for b in transform_batch:
                generative_kwargs = dict(transform_batch=b)
                inference_kwargs = dict(n_samples=n_samples)
                _, generative_outputs = self.module.forward(
                    tensors=tensors,
                    inference_kwargs=inference_kwargs,
                    generative_kwargs=generative_kwargs,
                    compute_loss=False,
                )
                py_mixing += torch.sigmoid(generative_outputs["py_"]["mixing"])[
                    ..., protein_mask
                ].cpu()
            py_mixing /= len(transform_batch)
            py_mixings += [py_mixing]
        if n_samples > 1:
            # concatenate along batch dimension -> result shape = (samples, cells, features)
            py_mixings = torch.cat(py_mixings, dim=1)
            # (cells, features, samples)
            py_mixings = py_mixings.permute(1, 2, 0)
        else:
            py_mixings = torch.cat(py_mixings, dim=0)

        if return_mean is True and n_samples > 1:
            py_mixings = torch.mean(py_mixings, dim=-1)

        py_mixings = py_mixings.cpu().numpy()

        if return_numpy is True:
            return 1 - py_mixings
        else:
            pro_names = self.protein_state_registry.column_names
            foreground_prob = pd.DataFrame(
                1 - py_mixings,
                columns=pro_names[protein_mask],
                index=adata.obs_names[indices],
            )
            return foreground_prob

    def _expression_for_de(
        self,
        adata=None,
        indices=None,
        n_samples_overall=None,
        transform_batch: Optional[Sequence[Union[Number, str]]] = None,
        scale_protein=False,
        batch_size: Optional[int] = None,
        sample_protein_mixing=False,
        include_protein_background=False,
        protein_prior_count=0.5,
    ):
        rna, protein = self.get_normalized_expression(
            adata=adata,
            indices=indices,
            n_samples_overall=n_samples_overall,
            transform_batch=transform_batch,
            return_numpy=True,
            n_samples=1,
            batch_size=batch_size,
            scale_protein=scale_protein,
            sample_protein_mixing=sample_protein_mixing,
            include_protein_background=include_protein_background,
        )
        protein += protein_prior_count

        joint = np.concatenate([rna, protein], axis=1)
        return joint

    @de_dsp.dedent
    def differential_expression(
        self,
        adata: Optional[AnnData] = None,
        groupby: Optional[str] = None,
        group1: Optional[Iterable[str]] = None,
        group2: Optional[str] = None,
        idx1: Optional[Union[Sequence[int], Sequence[bool], str]] = None,
        idx2: Optional[Union[Sequence[int], Sequence[bool], str]] = None,
        mode: Literal["vanilla", "change"] = "change",
        delta: float = 0.25,
        batch_size: Optional[int] = None,
        all_stats: bool = True,
        batch_correction: bool = False,
        batchid1: Optional[Iterable[str]] = None,
        batchid2: Optional[Iterable[str]] = None,
        fdr_target: float = 0.05,
        silent: bool = False,
        protein_prior_count: float = 0.1,
        scale_protein: bool = False,
        sample_protein_mixing: bool = False,
        include_protein_background: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        r"""A unified method for differential expression analysis.

        Implements `"vanilla"` DE :cite:p:`Lopez18`. and `"change"` mode DE :cite:p:`Boyeau19`.

        Parameters
        ----------
        %(de_adata)s
        %(de_groupby)s
        %(de_group1)s
        %(de_group2)s
        %(de_idx1)s
        %(de_idx2)s
        %(de_mode)s
        %(de_delta)s
        %(de_batch_size)s
        %(de_all_stats)s
        %(de_batch_correction)s
        %(de_batchid1)s
        %(de_batchid2)s
        %(de_fdr_target)s
        %(de_silent)s
        protein_prior_count
            Prior count added to protein expression before LFC computation
        scale_protein
            Force protein values to sum to one in every single cell (post-hoc normalization)
        sample_protein_mixing
            Sample the protein mixture component, i.e., use the parameter to sample a Bernoulli
            that determines if expression is from foreground/background.
        include_protein_background
            Include the protein background component as part of the protein expression
        use_field
            By default uses protein and RNA field disable here to perform only RNA or protein DE.
        pseudocounts
            pseudocount offset used for the mode `change`.
            When None, observations from non-expressed genes are used to estimate its value.
        **kwargs
            Keyword args for :meth:`scvi.model.base.DifferentialComputation.get_bayes_factors`

        Returns
        -------
        Differential expression DataFrame.
        """


        adata = self._validate_anndata(adata)
        adata_manager = self.get_anndata_manager(adata, required=True)
        model_fn = partial(
            self._expression_for_de,
            scale_protein=scale_protein,
            sample_protein_mixing=sample_protein_mixing,
            include_protein_background=include_protein_background,
            protein_prior_count=protein_prior_count,
            batch_size=batch_size,
        )
        col_names = np.concatenate(
            [
                np.asarray(_get_var_names_from_manager(adata_manager)),
                self.protein_state_registry.column_names,
            ]
        )
        result = _de_core(
            adata_manager,
            model_fn,
            groupby,
            group1,
            group2,
            idx1,
            idx2,
            all_stats,
            cite_seq_raw_counts_properties,
            col_names,
            mode,
            batchid1,
            batchid2,
            delta,
            batch_correction,
            fdr_target,
            silent,
            **kwargs,
        )

        return result

    @torch.inference_mode()
    def posterior_predictive_sample(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        n_samples: int = 1,
        batch_size: Optional[int] = None,
        gene_list: Optional[Sequence[str]] = None,
        protein_list: Optional[Sequence[str]] = None,
        atac_list: Optional[Sequence[str]] = None,
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

        if atac_list is None:
            atac_mask = slice(None)
        else:
            all_atac = self.atac_state_registry.column_names
            atac_mask = [True if atac in atac_list else False for atac in all_atac]

        scdl = self._make_data_loader(  # Need to implement in hierarVAE to return atac in scdl. Currently inherited from totalVAE in scvi code
            adata=adata, indices=indices, batch_size=batch_size
        )

        scdl_list = []
        for tensors in scdl:
            rna_sample, protein_sample = self.module.sample(
                tensors, n_samples=n_samples, swap_latent=swap_latent
            )
            rna_sample = rna_sample[..., gene_mask]
            protein_sample = protein_sample[..., protein_mask]
            atac_sample = protein_sample[..., atac_mask]
            data = torch.cat([rna_sample, protein_sample, atac_sample], dim=-1).numpy()

            scdl_list += [data]
            if n_samples > 1:
                scdl_list[-1] = np.transpose(scdl_list[-1], (1, 2, 0))
        scdl_list = np.concatenate(scdl_list, axis=0)

        return scdl_list

    @torch.inference_mode()
    def _get_denoised_samples(
        self,
        adata=None,
        indices=None,
        n_samples: int = 25,
        batch_size: int = 64,
        rna_size_factor: int = 1000,
        transform_batch: Optional[int] = None,
    ) -> np.ndarray:
        """
        Return samples from an adjusted posterior predictive.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            indices of `adata` to use
        n_samples
            How may samples per cell
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        rna_size_factor
            size factor for RNA prior to sampling gamma distribution
        transform_batch
            int of which batch to condition on for all cells
        """
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        scdl_list = []
        for tensors in scdl:
            x = tensors[REGISTRY_KEYS.X_KEY]
            y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]

            generative_kwargs = dict(transform_batch=transform_batch)
            inference_kwargs = dict(n_samples=n_samples)
            with torch.inference_mode():
                (
                    inference_outputs,
                    generative_outputs,
                ) = self.module.forward(
                    tensors,
                    inference_kwargs=inference_kwargs,
                    generative_kwargs=generative_kwargs,
                    compute_loss=False,
                )
            px_ = generative_outputs["px_"]
            py_ = generative_outputs["py_"]
            device = px_["r"].device

            pi = 1 / (1 + torch.exp(-py_["mixing"]))
            mixing_sample = torch.distributions.Bernoulli(pi).sample()
            protein_rate = py_["rate_fore"]
            rate = torch.cat((rna_size_factor * px_["scale"], protein_rate), dim=-1)
            if len(px_["r"].size()) == 2:
                px_dispersion = px_["r"]
            else:
                px_dispersion = torch.ones_like(x).to(device) * px_["r"]
            if len(py_["r"].size()) == 2:
                py_dispersion = py_["r"]
            else:
                py_dispersion = torch.ones_like(y).to(device) * py_["r"]

            dispersion = torch.cat((px_dispersion, py_dispersion), dim=-1)

            # This gamma is really l*w using scVI manuscript notation
            p = rate / (rate + dispersion)
            r = dispersion
            l_train = torch.distributions.Gamma(r, (1 - p) / p).sample()
            data = l_train.cpu().numpy()
            # make background 0
            data[:, :, x.shape[1] :] = data[:, :, x.shape[1] :] * (1 - mixing_sample).cpu().numpy()
            scdl_list += [data]

            scdl_list[-1] = np.transpose(scdl_list[-1], (1, 2, 0))

        return np.concatenate(scdl_list, axis=0)

    @torch.inference_mode()
    def get_feature_correlation_matrix(
        self,
        adata=None,
        indices=None,
        n_samples: int = 10,
        batch_size: int = 64,
        rna_size_factor: int = 1000,
        transform_batch: Optional[Sequence[Union[Number, str]]] = None,
        correlation_type: Literal["spearman", "pearson"] = "spearman",
        log_transform: bool = False,
    ) -> pd.DataFrame:
        """
        Generate gene-gene correlation matrix using scvi uncertainty and expression.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        rna_size_factor
            size factor for RNA prior to sampling gamma distribution
        transform_batch
            Batches to condition on.
            If transform_batch is:

            - None, then real observed batch is used
            - int, then batch transform_batch is used
            - list of int, then values are averaged over provided batches.
        correlation_type
            One of "pearson", "spearman".
        log_transform
            Whether to log transform denoised values prior to correlation calculation.

        Returns
        -------
        Gene-protein-gene-protein correlation matrix
        """
        from scipy.stats import spearmanr

        adata = self._validate_anndata(adata)
        adata_manager = self.get_anndata_manager(adata, required=True)

        if not isinstance(transform_batch, IterableClass):
            transform_batch = [transform_batch]

        transform_batch = _get_batch_code_from_category(
            self.get_anndata_manager(adata, required=True), transform_batch
        )

        corr_mats = []
        for b in transform_batch:
            denoised_data = self._get_denoised_samples(
                n_samples=n_samples,
                batch_size=batch_size,
                rna_size_factor=rna_size_factor,
                transform_batch=b,
            )
            flattened = np.zeros((denoised_data.shape[0] * n_samples, denoised_data.shape[1]))
            for i in range(n_samples):
                flattened[denoised_data.shape[0] * (i) : denoised_data.shape[0] * (i + 1)] = (
                    denoised_data[:, :, i]
                )
            if log_transform is True:
                flattened[:, : self.n_genes] = np.log(flattened[:, : self.n_genes] + 1e-8)
                flattened[:, self.n_genes :] = np.log1p(flattened[:, self.n_genes :])
            if correlation_type == "pearson":
                corr_matrix = np.corrcoef(flattened, rowvar=False)
            else:
                corr_matrix, _ = spearmanr(flattened, axis=0)
            corr_mats.append(corr_matrix)

        corr_matrix = np.mean(np.stack(corr_mats), axis=0)
        var_names = _get_var_names_from_manager(adata_manager)
        names = np.concatenate(
            [
                np.asarray(var_names),
                self.protein_state_registry.column_names,
            ]
        )
        return pd.DataFrame(corr_matrix, index=names, columns=names)

    @torch.inference_mode()
    def get_likelihood_parameters(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        n_samples: Optional[int] = 1,
        give_mean: Optional[bool] = False,
        batch_size: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        r"""
        Estimates for the parameters of the likelihood :math:`p(x, y \mid z)`.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        n_samples
            Number of posterior samples to use for estimation.
        give_mean
            Return expected value of parameters or a samples
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        """
        raise NotImplementedError

    def _validate_anndata(self, adata: Optional[AnnData] = None, copy_if_view: bool = True):
        adata = super()._validate_anndata(adata=adata, copy_if_view=copy_if_view)
        error_msg = "Number of {} in anndata different from when setup_anndata was run. Please rerun setup_anndata."
        if REGISTRY_KEYS.PROTEIN_EXP_KEY in self.adata_manager.data_registry.keys():
            pro_exp = self.get_from_registry(adata, REGISTRY_KEYS.PROTEIN_EXP_KEY)
            if self.summary_stats.n_proteins != pro_exp.shape[1]:
                raise ValueError(error_msg.format("proteins"))
            is_nonneg_int = _check_nonnegative_integers(pro_exp)
            if not is_nonneg_int:
                warnings.warn(
                    "Make sure the registered protein expression in anndata contains unnormalized count data."
                )
        # else:
        #     raise ValueError(
        #         "No protein data found, please setup or transfer anndata")

        return adata

    def _get_totalvi_protein_priors(self, adata, n_cells=100):
        """Compute an empirical prior for protein background."""
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.mixture import GaussianMixture

        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            logger.info("Computing empirical prior initialization for protein background.")

            adata = self._validate_anndata(adata)
            adata_manager = self.get_anndata_manager(adata)
            pro_exp = adata_manager.get_from_registry(REGISTRY_KEYS.PROTEIN_EXP_KEY)
            pro_exp = pro_exp.to_numpy() if isinstance(pro_exp, pd.DataFrame) else pro_exp
            batch_mask = adata_manager.get_state_registry(REGISTRY_KEYS.PROTEIN_EXP_KEY).get(
                fields.ProteinObsmField.PROTEIN_BATCH_MASK
            )
            batch = adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY).ravel()
            cats = adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY)[
                fields.CategoricalObsField.CATEGORICAL_MAPPING_KEY
            ]
            codes = np.arange(len(cats))

            batch_avg_mus, batch_avg_scales = [], []
            for b in np.unique(codes):
                # can happen during online updates
                # the values of these batches will not be used
                num_in_batch = np.sum(batch == b)
                if num_in_batch == 0:
                    batch_avg_mus.append(0)
                    batch_avg_scales.append(1)
                    continue
                batch_pro_exp = pro_exp[batch == b]

                # non missing
                if batch_mask is not None:
                    batch_pro_exp = batch_pro_exp[:, batch_mask[str(b)]]
                    if batch_pro_exp.shape[1] < 5:
                        logger.debug(
                            f"Batch {b} has too few proteins to set prior, setting randomly."
                        )
                        batch_avg_mus.append(0.0)
                        batch_avg_scales.append(0.05)
                        continue

                # a batch is missing because it's in the reference but not query data
                # for scarches case, these values will be replaced by original state dict
                if batch_pro_exp.shape[0] == 0:
                    batch_avg_mus.append(0.0)
                    batch_avg_scales.append(0.05)
                    continue

                cells = np.random.choice(np.arange(batch_pro_exp.shape[0]), size=n_cells)
                batch_pro_exp = batch_pro_exp[cells]
                gmm = GaussianMixture(n_components=2)
                mus, scales = [], []
                # fit per cell GMM
                for c in batch_pro_exp:
                    try:
                        gmm.fit(np.log1p(c.reshape(-1, 1)))
                    # when cell is all 0
                    except ConvergenceWarning:
                        mus.append(0)
                        scales.append(0.05)
                        continue

                    means = gmm.means_.ravel()
                    sorted_fg_bg = np.argsort(means)
                    mu = means[sorted_fg_bg].ravel()[0]
                    covariances = gmm.covariances_[sorted_fg_bg].ravel()[0]
                    scale = np.sqrt(covariances)
                    mus.append(mu)
                    scales.append(scale)

                # average distribution over cells
                batch_avg_mu = np.mean(mus)
                batch_avg_scale = np.sqrt(np.sum(np.square(scales)) / (n_cells**2))

                batch_avg_mus.append(batch_avg_mu)
                batch_avg_scales.append(batch_avg_scale)

            # repeat prior for each protein
            batch_avg_mus = np.array(batch_avg_mus, dtype=np.float32).reshape(1, -1)
            batch_avg_scales = np.array(batch_avg_scales, dtype=np.float32).reshape(1, -1)
            batch_avg_mus = np.tile(batch_avg_mus, (pro_exp.shape[1], 1))
            batch_avg_scales = np.tile(batch_avg_scales, (pro_exp.shape[1], 1))

        return batch_avg_mus, batch_avg_scales

    @torch.inference_mode()
    def get_latent_representation(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        give_mean: bool = True,
        mc_samples: int = 5000,
        batch_size: Optional[int] = None,
        return_dist: bool = False,
        add_latents_to_adata: bool = True,
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
        scdl = self._make_data_loader(  # Need to implement in hierarVAE to return atac in scdl. Currently inherited from totalVAE in scvi code
            adata=adata, indices=indices, batch_size=batch_size
        )
        latent1 = []
        latent2 = []
        latent1r = []
        latent1p = []
        latent1a = []

        for tensors in scdl:
            # Need to implement in hierarVAE to return atac in inference_inputs. Currently inherited from totalVAE
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            if "qz1" in outputs:
                qz1 = outputs["qz1"]
                qz2 = outputs["qz2"]
                qz1r = outputs["qz1r"]
                qz1p = outputs["qz1p"]
                qz1a = outputs["qz1a"]
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
                    z1p = qz1p.loc if qz1p is not None else None
                    z1a = qz1a.loc if qz1a is not None else None

            else:
                z1 = outputs["z1"]
                z2 = outputs["z2"]
                z1r = outputs["z1r"]
                z1p = outputs["z1p"]
                z1a = outputs["z1a"]

            latent1 += [z1.cpu()]
            latent2 += [z2.cpu()]
            latent1r += [z1r.cpu()]
            if z1p is not None:
                latent1p += [z1p.cpu()]
            if z1a is not None:
                latent1a += [z1a.cpu()]

        latent1 = torch.cat(latent1).numpy()
        latent2 = torch.cat(latent2).numpy()
        latent1r = torch.cat(latent1r).numpy()
        latent1p = torch.cat(latent1p).numpy() if latent1p != [] else None
        latent1a = torch.cat(latent1a).numpy() if latent1a != [] else None
        latent_c = np.concatenate((latent1, latent1r), axis=1)
        if latent1p is not None:
            latent_c = np.concatenate((latent_c, latent1p), axis=1)
        if latent1a is not None:
            latent_c = np.concatenate((latent_c, latent1a), axis=1)
        if add_latents_to_adata:
            adata.obsm["Z_multiHIVE"] = latent_c
            adata.obsm["Z1_multiHIVE"] = latent1
            adata.obsm["Z2_multiHIVE"] = latent2
            adata.obsm["Zr_multiHIVE"] = latent1r
            if latent1p is not None:
                adata.obsm["Zp_multiHIVE"] = latent1p
            if latent1a is not None:
                adata.obsm["Za_multiHIVE"] = latent1a
            return

        return dict(
            Z_multiHIVE=latent_c,
            Z1_multiHIVE=latent1,
            Z2_multiHIVE=latent2,
            Zr_multiHIVE=latent1r,
            Zp_multiHIVE=latent1p,
            Za_multiHIVE=latent1a,
        )

    @torch.inference_mode()
    def get_accessibility_estimates(
        self,
        adata: Optional[AnnData] = None,
        indices: Sequence[int] = None,
        n_samples_overall: Optional[int] = None,
        region_list: Optional[Sequence[str]] = None,
        transform_batch: Optional[Union[str, int]] = None,
        use_z_mean: bool = True,
        threshold: Optional[float] = None,
        normalize_cells: bool = False,
        normalize_regions: bool = False,
        batch_size: int = 128,
        return_numpy: bool = False,
    ) -> Union[np.ndarray, csr_matrix, pd.DataFrame]:
        """
        Impute the full accessibility matrix.

        Returns a matrix of accessibility probabilities for each cell and genomic region in the input
        (for return matrix A, A[i,j] is the probability that region j is accessible in cell i).

        Parameters
        ----------
        adata
            AnnData object that has been registered with scvi. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        n_samples_overall
            Number of samples to return in total
        region_indices
            Indices of regions to use. if `None`, all regions are used.
        transform_batch
            Batch to condition on.
            If transform_batch is:

            - None, then real observed batch is used
            - int, then batch transform_batch is used
        use_z_mean
            If True (default), use the distribution mean. Otherwise, sample from the distribution.
        threshold
            If provided, values below the threshold are replaced with 0 and a sparse matrix
            is returned instead. This is recommended for very large matrices. Must be between 0 and 1.
        normalize_cells
            Whether to reintroduce library size factors to scale the normalized probabilities.
            This makes the estimates closer to the input, but removes the library size correction.
            False by default.
        normalize_regions
            Whether to reintroduce region factors to scale the normalized probabilities. This makes
            the estimates closer to the input, but removes the region-level bias correction. False by
            default.
        batch_size
            Minibatch size for data loading into model
        """
        self._check_adata_modality_weights(adata)
        adata = self._validate_anndata(adata)
        adata_manager = self.get_anndata_manager(adata, required=True)
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        post = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        transform_batch = _get_batch_code_from_category(adata_manager, transform_batch)

        if region_list is None:
            region_mask = slice(None)
        else:
            region_mask = [region in region_list for region in adata.var_names[self.n_genes :]]

        if threshold is not None and (threshold < 0 or threshold > 1):
            raise ValueError("the provided threshold must be between 0 and 1")

        imputed = []
        for tensors in post:
            get_generative_input_kwargs = dict(transform_batch=transform_batch[0])
            generative_kwargs = dict(use_z_mean=use_z_mean)
            inference_outputs, generative_outputs = self.module.forward(
                tensors=tensors,
                get_generative_input_kwargs=get_generative_input_kwargs,
                generative_kwargs=generative_kwargs,
                compute_loss=False,
            )
            p = generative_outputs["p"].cpu()

            if normalize_cells:
                p *= inference_outputs["libsize_acc"].cpu()
            if normalize_regions:
                p *= torch.sigmoid(self.module.region_factors).cpu()
            if threshold:
                p[p < threshold] = 0
                p = csr_matrix(p.numpy())
            if region_mask is not None:
                p = p[:, region_mask]
            imputed.append(p)

        if threshold:  # imputed is a list of csr_matrix objects
            imputed = vstack(imputed, format="csr")
        else:  # imputed is a list of tensors
            imputed = torch.cat(imputed).numpy()

        if return_numpy:
            return imputed
        elif threshold:
            return pd.DataFrame.sparse.from_spmatrix(
                imputed,
                index=adata.obs_names[indices],
                columns=adata.var_names[self.n_genes :][region_mask],
            )
        else:
            return pd.DataFrame(
                imputed,
                index=adata.obs_names[indices],
                columns=adata.var_names[self.n_genes :][region_mask],
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

        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

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

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: str | None = None,
        batch_key: str | None = None,
        size_factor_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        protein_expression_obsm_key: str | None = None,
        protein_names_uns_key: str | None = None,
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_layer)s
        %(param_batch_key)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        protein_expression_obsm_key
            key in `adata.obsm` for protein expression data.
        protein_names_uns_key
            key in `adata.uns` for protein names. If None, will use the column names of
            `adata.obsm[protein_expression_obsm_key]` if it is a DataFrame, else will assign
            sequential names to proteins.
        """
        warnings.warn(
            "multiHIVE is supposed to work with MuData. the use of anndata is "
            "deprecated and will be removed in scvi-tools 1.4. Please use setup_mudata",
            DeprecationWarning,
            stacklevel=settings.warnings_stacklevel,
        )
        setup_method_args = cls._get_setup_method_args(**locals())
        adata.obs["_indices"] = np.arange(adata.n_obs)
        batch_field = CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key)
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            batch_field,
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, None),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            NumericalJointObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
            NumericalObsField(REGISTRY_KEYS.INDICES_KEY, "_indices"),
        ]
        if protein_expression_obsm_key is not None:
            anndata_fields.append(
                ProteinObsmField(
                    REGISTRY_KEYS.PROTEIN_EXP_KEY,
                    protein_expression_obsm_key,
                    use_batch_mask=True,
                    batch_field=batch_field,
                    colnames_uns_key=protein_names_uns_key,
                    is_count_data=True,
                )
            )

        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
