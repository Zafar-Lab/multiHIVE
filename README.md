# MultiHIVE

## Basic Installation

To install the required dependencies, run the following commands:

```bash
pip install scvi-tools
pip install matplotlib
pip install scikit-learn
```

## Steps to Run

Run the `user.py` script to execute the model:

```bash
python3 user.py
```

## Folder Structure and Parameters

1. **Main Script**:

   - `user.py` is the main script used to run the model.

2. **Model Parameters**:

   - ``: Determines the probability distribution of the latent space (e.g., `Normal`).
   - ``: Used for regularization of modality-specific latent distributions. Set to `True`.
   - ``: Enables multiple layers in the underlying neural network. Set to `True` if the dataset contains more than **100,000 cells**.

3. **Additional Resources**:

   - Refer to the [scvi-tools documentation](https://github.com/scverse/scvi-tools) for details on preprocessing parameters and other advanced configurations.

