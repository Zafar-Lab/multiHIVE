.. multiHIVE documentation master file, created by
   sphinx-quickstart on Tue Jul  1 15:37:07 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Overview
========

Welcome to multiHIVE's documentation!
-------------------------------------

**multiHIVE** multiHIVE, a hierarchical multimodal deep generative model for inferring cellular embeddings by integrating CITE-seq data modalities. multiHIVE employs hierarchically stacked latent variables as well as modality-specific latent variables to capture shared and private information from the modalities respectively, facilitating integration, denoising and protein imputation and integration of CITE-seq dataset with unimodal dataset. The factorization of multiHIVE-inferred denoised expression into gene expression programs aids in identifying biological processes at multiple levels of cellular hierarchy.

.. image:: ../../Architecture.png





Cite this article
-----------------
multiHIVE: Hierarchical Multimodal Deep Generative Model for Single-cell Multiomics Integration  
Anirudh Nanduri\*, Musale Krushna Pavan\*, Kushagra Pandey, Hamim Zafar  
bioRxiv 2025.01.28.635222; doi: https://doi.org/10.1101/2025.01.28.635222  
\*Equal contribution


.. toctree::
   :maxdepth: 2

   self
   installation
   tutorials
   multiHIVE
   multiHIVEvae
   base_components
   references
                  
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`