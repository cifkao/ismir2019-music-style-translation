# Supervised symbolic music style translation
This is the code for the paper [‘Supervised symbolic music style translation using synthetic data’](https://doi.org/10.5281/zenodo.3527878), accepted to ISMIR 2019. If you use the code in your research, please cite the paper as:

> Ondřej Cífka, Umut Şimşekli, Gaël Richard. “Supervised Symbolic Music Style Translation Using Synthetic Data”, *20th International Society for Music Information Retrieval Conference*, Delft, The Netherlands, 2019. [doi:10.5281/zenodo.3527878](https://doi.org/10.5281/zenodo.3527878).

Also, check out the [example outputs](http://tiny.cc/musicstyle) and the accompanying [blog post](https://ondrej.cifka.com/posts/neural-music-style-translation/), which summarizes the paper.

The repository contains the following directories:
- `code` – code for training and evaluating models
- `experiments` – configuration files for the models from the paper
- `data` – data preparation recipes

You can either [download](https://doi.org/10.5281/zenodo.3245374) the trained models, or train your own by following the steps below. If you encounter any problems, please feel free to [open an issue](https://github.com/cifkao/ismir2019-music-style-translation/issues).

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
   pip install './code[gpu]'
   ```
   Or for the non-GPU version (only if you skipped step 1):
   ```sh
   pip install './code[nogpu]'
   ```

## Data

See the [data README](data/README.md) for how to prepare the data.

## Training a model

The scripts for training the models are in the `ismir2019_cifka.models` package.

The [`experiments`](experiments) directory has a subdirectory for each model from the paper. The `model.yaml` file in each directory contains all the hyperparameters and other settings required to train and use the model; the first line also tells you what type of model it is (i.e. `seq2seq_style` or `roll2seq_style`).  For example, to train the `all2bass` model, run the following command inside the `experiments` directory:
```sh
python -m ismir2019_cifka.models.roll2seq_style --logdir all2bass train
```
You may need to adjust the paths in `model.yaml` to point to your dataset.

## Running a model

Before running a trained model on some MIDI files, we need to use the `chop_midi` script to chop them up into segments and save them in the expected format (see the [data README](data/README.md) for more information), e.g.:
```sh
python -m ismir2019_cifka.data.chop_midi \
    --no-drums \
    --force-tempo 60 \
    --bars-per-segment 8 \
    --include-segment-id \
    song1.mid song2.mid songs.pickle
```
Then we can `run` the model, providing the input file, the output file and the target style. For example:
```sh
python -m ismir2019_cifka.models.roll2seq_style --logdir all2bass run songs.pickle output.pickle ZZREGGAE
```
To listen to the outputs, we need to convert them back to MIDI files, which involves time-stretching the music from 60 BPM to the desired tempo, assigning an instrument, and concatenating the segments of each song:
```sh
python -m ismir2019_cifka.data.notes2midi \
   --instrument 'Fretless Bass' \
   --stretch 60:115 \
   --group-by-name \
   --time-unit 4 \
   output.pickle outputs
```

## Evaluation

To reproduce the results on the Bodhidharma dataset, first [download the trained models](https://doi.org/10.5281/zenodo.3245374) and [prepare the dataset](data/README.md), then change to the `experiments` directory and run `./evaluate_bodhidharma.sh`. Note that this will run each model many times on the entire dataset (once for each target style), so you might want to start with only a subset of the models or styles or run a number of them in parallel. The results will be stored in the `results` subdirectory; use the [`evaluation.ipynb`](experiments/evaluation.ipynb) Jupyter notebook to load and plot them.

To compute the metrics on your own data, use `python -m ismir2019_cifka.evaluate` directly. To better understand all the arguments, see how they are used in [`evaluate_bodhidharma.sh`](experiments/evaluate_bodhidharma.sh). The tricky ones are:

* `--data-prefix`: where to look for the model outputs inside the model directory; for example, if you pass `--data-prefix outputs/test_`, then the outputs of model `model1` in style `A` will be taken from `model1/outputs/test_A.pickle`
* `--style-profile-dir`: a directory containing JSON files with reference style profiles; you can generate these using `python -m ismir2019_cifka.eval.style_profile`

Alternatively, you can import the evaluation metrics from the `ismir2019_cifka.eval` package and use them from your own code.

## Acknowledgment
This work has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No. 765068.

## Copyright notice
Copyright 2019 Ondřej Cífka of Télécom Paris, Institut Polytechnique de Paris.  
All rights reserved.
