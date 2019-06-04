# Supervised symbolic music style translation
This is the code for the paper ‘Supervised symbolic music style translation using synthetic data’.

The repository contains the following directories:
- `cifka2019` – code for training and evaluating models.
- `experiments` – configuration files for the models from the paper.
- `data` – data preparation pipeline.

## Installation

Clone the repository and make sure you have Python 3.6 or later. Then run the following commands.

1. (optional) To make sure you have the right versions of the most important packages, run:
   ```sh
   pip install -r requirements.txt
   ```
   Alternatively, if you use conda, you can create your environment using
   ```sh
   conda env create -f environment.yml
   ```
   This will also install the correct versions of the CUDA and CuDNN libraries.
   
   If you wish to use different (more recent) package versions, you may skip this step; the code should still work.

2. Install the package with:

   ```sh
   pip install .
   ```

## Data

See the [data README](data/README.md) for how to prepare the data.

The data for each track is stored in numbered pickle files. In the paper, we used files 00–54 for training, 55 for validation and 56 for testing.
