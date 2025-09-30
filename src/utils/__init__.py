import itertools
import logging
import pathlib
import random
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display as disp

import torch
from pydub import AudioSegment

import utils
from neuralNetwork import DataPreprocessing as DataPreprocseeing
from pipelineConfig.PipelineConfigLoader import PipelineConfigLoader
from trainingData import trainingDatabase
from trainingData.trainingDatabase import TrainingDatabaseConnection

_DO_EXPORT_SYNTHESIS_DATA = "do_export_synthesis_data"
_DATA_PATH = "data_path"
_SAMPLE_WIDTH = "sample_width"
_CHANNELS = "channels"
_SAMPLE_RATE = "sample_rate_hz"
_CHECKPOINT_PATH_KEY = "checkpoint_path"
_CHECKPOINT_TO_LOAD_KEY = "load_checkpoint"
_MAX_NUMBER_OF_SAMPLES_KEY = "max_number_of_samples"
_NONE_CATEGORY_NAME = "none"


def export_samples_as_mp3(samples, config):
    """
    Exports the given samples to the given path as mp3.
    :param samples: The samples.
    :param config: The export config.
    :return: None.
    """
    # Check whether export is active or there are no samples
    if not config[_DO_EXPORT_SYNTHESIS_DATA] or not samples:
        # Nothing to do
        return
    # Get the path
    path = pathlib.Path(config[_DATA_PATH])
    # Check if this exists and if so whether it is a directory.
    if path.exists():
        # Exists but is not a directory
        if not path.is_dir():
            raise ValueError(
                f"The sample export path path '{path}' is not a directory."
            )
    else:
        # Create the directory with the parents in case they do not exist
        path.mkdir(parents=True)
    # Iterate over the samples
    for i, sample in enumerate(samples):
        # Create the pydub audio file, transforming the numpy array to bytes
        audio_file = AudioSegment(
            data=sample.data.tobytes(),
            sample_width=config[_SAMPLE_WIDTH],
            frame_rate=config[_SAMPLE_RATE],
            channels=config[_CHANNELS],
        )
        # Export the audio file as mp3
        audio_file.export(path / f"sample_{i}.mp3", format="mp3")


def get_all_subclasses(cls) -> set:
    """
    Recursively gets and returns all subclasses of the given class.

    E.g.:
    Test <- Test2 <- Test4
    ^
    I
    Test3

    get_all_subclasses(Test) == {Test2, Test4, Test3}

    The classes have to have been imported once to be in the classpath.

    :param cls: The class to get the subclasses of.

    :return: The subclasses of the given class.
    """
    return set(
        itertools.chain(
            *(
                (subclass, *get_all_subclasses(subclass))
                for subclass in cls.__subclasses__()
            )
        )
    )


def init_checkpointing(config: PipelineConfigLoader, model_name: str, cpu=False):
    """
    Checks that the checkpoint dir exists and loads a previously saved checkpoint if configured.

    :param config: The configuration to use for the training.
    :param model_name The name of the model
    :param cpu Force using cpu
    :return: The checkpointing object.
    """
    # if cuda is available use GPU to train the model
    device = torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')

    checkpoint_path = pathlib.Path(config.train_config[_CHECKPOINT_PATH_KEY] + '/' + model_name)
    # Check if the checkpoint dir exists
    if not checkpoint_path.exists():
        # Create it with parents
        checkpoint_path.mkdir(parents=True)
        # Log directory
        logging.info(f"Created checkpoint directory: {checkpoint_path}")
    logging.info(f"Will use checkpoint directory: {checkpoint_path}")
    # Get the path of the checkpoint file
    checkpoint_to_load = checkpoint_path / config.train_config[_CHECKPOINT_TO_LOAD_KEY]
    # Check if the checkpoint file exists
    if not checkpoint_to_load.exists():
        # Log
        logging.warning(
            f"No checkpoint found at {checkpoint_to_load}, will use random initialization."
        )
        logging.warning("(Can be ignored if you are starting from scratch)")
        # Checkpoint does not exist, return None
        return None
    # Log checkpoint loading
    logging.info(f"Will load checkpoint from: {checkpoint_to_load}")
    # Checkpoint exists, load it
    return torch.load(checkpoint_to_load, map_location=device)


def get_category_index(config: PipelineConfigLoader, category_name: str) -> int:
    """
    Maps the given category name to the category id.

    :param config: The configuration to use.
    :param category_name: The name of the category.
    :return: The category id.
    """
    try:
        # Get the category index
        return list(config.input_data_config).index(category_name)
    except ValueError:
        # Category not found
        if category_name == _NONE_CATEGORY_NAME:
            # None category is the last index
            return len(config.input_data_config)
        raise ValueError(f"Category '{category_name}' not found in config.")


def get_category_name(config: PipelineConfigLoader, category_index: int) -> str:
    """
    Maps the given category id to the category name.

    :param config: The configuration to use.
    :param category_index: The category index.
    :return: The category name.
    """
    try:
        # Get the category name
        return list(config.input_data_config)[category_index]
    except IndexError:
        # Category not found
        if category_index == len(config.input_data_config):
            # None category is the last index
            return _NONE_CATEGORY_NAME
        raise ValueError(f"Category index {category_index} not found in config.")


def perform_preprocessing(config, samples, visualize=False, cpu=False):
    """
    Performs the preprocessing of the samples.
    :param config: The config to use.
    :param samples: The samples to preprocess. Note that the number of returned samples might differ
    from the number of samples given, if the number of samples is larger than the max number of samples as
    given by the config.
    :param visualize Param to visualize the preprocessed data
    :param cpu Force using cpu for preprocessing
    :return: The quantized samples, the labels and the number of unique labels.
    """
    # if cuda is available use GPU to train the model
    device = torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')
    # Check if there are more samples than the ones we want to train on
    if (
            config.train_config[_MAX_NUMBER_OF_SAMPLES_KEY] != -1
            and len(samples) > config.train_config[_MAX_NUMBER_OF_SAMPLES_KEY]
    ):
        # If so, sample a random subset of the samples
        samples = random.sample(
            samples, config.train_config[_MAX_NUMBER_OF_SAMPLES_KEY]
        )
    if visualize:
        plot_librosa(samples[0])
    # Perform preprocessing
    preprocessed_samples = [
        show_progress(DataPreprocseeing.py_speech_preprocessing(sample.data, sample.sample_rate),
                      i, len(samples), len(samples) / 10)
        for i, sample in enumerate(samples)
    ]
    # Quantize the data
    quantized_samples = [
        torch.tensor(DataPreprocseeing.quantize_input(sample), device=device)
        for sample in preprocessed_samples
    ]
    # Stack the quantized samples
    quantized_samples = torch.vstack(quantized_samples)

    if visualize:
        data_to_show = preprocessed_samples.copy()
        data_to_show.extend([np.array([quantized_sample.cpu().numpy()]) for quantized_sample in quantized_samples])
        plot_spec(data_to_show, rows=2,
                  title=['preprocessed', 'quantized/reshaped Spec'], y_label=['Coefficients', ''], x_label=['Frame', ''],
                  x_tick=[lambda spec: np.linspace(0, 200, num=200//10 + 1),
                          lambda spec: np.linspace(0, spec.shape[1], num=11)],
                  block=True)
    # Zip together with labels
    labels = [
        ("none" if not sample.category else sample.category[0].name)
        for sample in samples
    ]
    number_of_unique_labels = len(set(labels))
    # Map the labels to the indexes
    labels = torch.tensor([utils.get_category_index(config, label) for label in labels], device=device)
    return quantized_samples, labels, number_of_unique_labels


def plot_librosa(sample):
    # calculate mel-spectrogram
    sample_spec = librosa.feature.melspectrogram(y=sample.data.astype(np.float32))
    # calculate mfcc
    sample_mfcc = librosa.feature.mfcc(y=sample.data.astype(np.float32), sr=sample.sample_rate)
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].set(title='mel Spec')
    # plot mel spec converted to db scale
    img = disp.specshow(librosa.amplitude_to_db(sample_spec), x_axis='time', y_axis='mel', ax=ax[0],
                        sr=sample.sample_rate)
    ax[1].set(title='mfcc Spec')
    # plot mfcc
    disp.specshow(sample_mfcc, ax=ax[1], sr=sample.sample_rate, x_axis='frames', y_axis=None)
    ax[1].set_ylabel('Coefficients')
    ax[1].set_yticks(np.linspace(0, 20, num=11))
    # show the plots with color
    plt.colorbar(img, ax=ax[0], format="%+2.f dB")


def plot_spec(spectrograms, rows=1, title=None, x_label=None, y_label=None, x_tick=None, y_tick=None, block=False):
    fig, axes = plt.subplots(rows, figsize=(10, 10))
    fig.tight_layout(pad=5)
    if rows == 1:
        # if only one row is wanted, treat the spectrogram as a single object
        spectrograms = [spectrograms]
    for i, spectrogram in enumerate(spectrograms):
        if i >= rows:
            break
        # get the axes object
        ax = axes if isinstance(axes, plt.Axes) else axes[i]
        if spectrogram.shape[0] < 2:
            # if the list/array has just one dimension, it is not displayed, so duplicate its content
            spectrogram = spectrogram.repeat(2, axis=0)
        if title is not None:
            ax.set_title(title if not type(title) == list else title[i])
        if x_label is not None:
            ax.set_xlabel(x_label if not type(x_label) == list else x_label[i])
        if y_label is not None:
            ax.set_ylabel(y_label if not type(y_label) == list else y_label[i])
        # set the x_tick(width of the spec)
        x = range(spectrogram.shape[1]) if x_tick is None else x_tick(spectrogram) if not type(
            x_tick) == list else x_tick[i](spectrogram)
        # set the y_tick(height of the spec)
        y = range(spectrogram.shape[0]) if y_tick is None else y_tick(spectrogram) if not type(
            y_tick) == list else y_tick[i](spectrogram)
        ax.pcolormesh(range(spectrogram.shape[1]), range(spectrogram.shape[0]), spectrogram)
        ax.set_yticks(y)
        ax.set_xticks(x)
    plt.show(block=block)


def show_progress(element, index, element_count, step):
    if index % step == 0:
        print(f'Progress {index}/{element_count}')
    return element
