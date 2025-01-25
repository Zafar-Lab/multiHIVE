MultuHIVE


Basic installation

pip install scvi-tools
pip install matplotlib
pip install scikit-learn


Steps to run:
python3 user.py

Folder structure:

1. user.py can be used to run the model.
2. Model takes following parameters:
    2.1 latent distribution - To determine the probability distribution of the normal space (Eg. Normal)
    2.2  kl_dot_product - Used for regularization of the modality specfic latent distributions. Set to true.
    2.3  deep_network - Uses multiple layers in the underlying neural network. Set to True if the dataset 
                        contains more than 100,000 cells.

3. Refer scvi https://github.com/scverse/scvi-tools for preprocessing parameters.
