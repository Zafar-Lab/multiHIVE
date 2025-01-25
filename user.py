from scvi.train import Trainer
from scvi.model import TOTALVI
import scanpy as sc
from matplotlib_inline.config import InlineBackend
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
import anndata as ad
import scvi
import matplotlib.pyplot as plt 

from src.model import HierarVI


def start_script():
    adata = sc.read('../notebooks/data/breast_cancer_cite.h5ad')
    adata.layers['counts'] = adata.X.copy()
    adata.var_names_make_unique()
    adata.obsm['protein_counts'] = adata.obsm['protein_counts'].toarray()
    import numpy as np
    #
    #
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=4000,
        flavor="seurat_v3",
        batch_key="batch",
        subset=True,
        layer="counts"
    )


    HierarVI.setup_anndata(
        adata,
        layer="counts",
        batch_key="batch",
        protein_expression_obsm_key="protein_counts"
    )


    vae = HierarVI(adata, latent_distribution="normal", kl_dot_product=True, deep_network=True)
    vae.train(max_epochs=200)
    adata.obsm["Z1_hierarVI"], adata.obsm["Z2_hierarVI"], adata.obsm["Z1r_hierarVI"], adata.obsm["Z1p_hierarVI"] = vae.get_latent_representation()
    adata.obsm['Zc_hierarVI'] = np.concatenate((adata.obsm["Z1_hierarVI"], adata.obsm["Z1r_hierarVI"], adata.obsm["Z1p_hierarVI"]), axis=1)

    #generated_data = vae.posterior_predictive_sample(adata, swap_latent=False)

    print(vae.history["elbo_train"])
    print(vae.history["elbo_validation"])
    plt.plot(vae.history["elbo_train"], label="train")
    plt.plot(vae.history["elbo_validation"], label="val")
    plt.title("Negative ELBO over training epochs")
    #plt.ylim(1100, 1500)
    plt.legend()





if __name__ == '__main__':
    start_script()
