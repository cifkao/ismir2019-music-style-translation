# Supervised symbolic music style translation
This is the code for the paper ‘Supervised symbolic music style translation using synthetic data’, accepted to ISMIR 2019.

The repository contains the following directories:
- `ismir2019_cifka` – code for training and evaluating models.
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

The data for each instrument track is stored in numbered pickle files. In the paper, we used files 00–54 for training, 55 for validation and 56 for testing.

## Training a model

The scripts for training the models are in the `ismir2019_cifka.models` package.

The `experiments` directory contains a subdirectory for each model from the paper. For example, to train the `all2bass` model, run the following command inside the `experiments` directory:
```sh
python -m ismir2019_cifka.models.roll2seq --logdir all2bass train
```
This will train the model using the `model.yaml` configuration file located in the model directory.

## Running a model

To use a trained model, run the same command as for training, but with `run` instead of `train`. You need to provide three additional arguments: the input file, the output file and the target style. For example:
```sh
python -m ismir2019_cifka.models.roll2seq --logdir all2bass run input.pickle output.pickle ZZREGGAE
```
To listen to the outputs, we need to convert them to MIDI files. This is an extra step that involves time-stretching the music from 60 BPM to the desired tempo, assigning an instrument, and concatenating the segments of each song:
```sh
python -m ismir2019_cifka.data.notes2midi \
   --instrument 'Fretless Bass' \
   --stretch 60:115 \
   --group-by-name \
   --time-unit 4 \
   output.pickle outputs
```

## Evaluation

## Acknowledgment
This work has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No. 765068.
