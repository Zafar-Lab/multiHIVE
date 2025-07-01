# multiHIVE

## Overview

**multiHIVE** is a hierarchical multimodal deep generative model designed to infer cellular embeddings by integrating multi-omics data with different modalities from the same cell. It uses:

- **Hierarchically stacked latent variables** to capture shared biological signals
- **Modality-specific latent variables** to model private (modality-unique) variation

This enables multiHIVE to perform:

- **Joint integration** of multi-modal data
- **Denoising**
- **Protein imputation**
- **Integration of multi-modal and uni-modal datasets**

Additionally, multiHIVE enables factorization of denoised gene expression into interpretable gene expression programs, facilitating the identification of biological processes at multiple levels of cellular hierarchy.

<p align="center">
  <img src="Architecture.png" alt="multiHIVE Architecture" width="500"/>
</p>

## Basic Installation

we recommend users to directly clone our stable main branch and set multiHIVE as the working directory and install following dependencies in a new conda environment `python>=3.11`

```bash
git clone https://github.com/Zafar-Lab/multiHIVE.git
pip install scvi-tools==1.3.0
pip install scanpy==1.11.0
pip install scikit-misc
```

## Or install directly via pip
```bash
pip install multiHIVE
```
## Tutorials
Explore the following tutorials to get started:  
- CITE-seq integration: [Tutorials/CITE_seq_Integration.ipynb](Tutorials/CITE_seq_Integration.ipynb)  
- Protein imputation: [Tutorials/Protein_Imputatoin.ipynb](Tutorials/Protein_Imputatoin.ipynb)  
- TEA-seq integration: [Tutorials/TEA-seq_integration.ipynb](Tutorials/Protein_Imputatoin.ipynb)


### 1. **Main Script**:

```
# features should genes followed by regions [genes, regions]
multiHIVE.setup_anndata(adata, batch_key="batch", protein_expression_obsm_key = "protein_expression")

vae = multiHIVE(adata,  
            n_genes=(adata.var["modality"] == "Gene Expression").sum(), # number of genes 
            n_regions=(adata.var["modality"] == "Peaks").sum(), # number of regions
            n_proteins=46, # number of proteins 
            latent_distribution="normal", kl_dot_product=True, deep_network=True)
vae.train()
vae.get_latent_representation()
```

  
### 2. **Model Parameters**:

| Parameter            | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `latent_distribution`| Distribution for latent variables (e.g., `"normal"`)                        |
| `kl_dot_product`     | Enables regularization using dot-product of modality-specific latents       |
| `deep_network`       | Uses deeper neural networks; recommended for datasets > 100,000 cells       |


### 3. **Results**:

   - vae.get_latent_representation() gives zs1, zs2, zr and zp or/and za
   -  zs1 is the joint latent variable. 
   -  zs2 is the hierarchical joint latent variable.
   -  zr is the gene modality specific latent variable.
   -  zp is the protein modality specific latent variable.
   -  za is the chromatin accessibility specific latent variable

## Documentation
For more advanced settings, preprocessing tips, and API references, refer to the multiHIVE Documentation (link coming soon)

## Citation
multiHIVE: Hierarchical Multimodal Deep Generative Model for Single-cell Multiomics Integration  
Anirudh Nanduri\*, Musale Krushna Pavan\*, Kushagra Pandey, Hamim Zafar  
bioRxiv 2025.01.28.635222; doi: https://doi.org/10.1101/2025.01.28.635222  
\*Equal contribution

## Contact
For questions, issues, or contributions, please open an issue on the [GitHub repository](https://github.com/Zafar-Lab/multiHIVE)  
