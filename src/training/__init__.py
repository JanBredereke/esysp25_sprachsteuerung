import itertools
import logging
import pathlib
from typing import Optional, List

import torch
import sys
import traceback

import utils
from neuralNetwork.NeuralNetwork import NeuralNetwork
from pipelineConfig.PipelineConfigLoader import PipelineConfigLoader
from trainingData import trainingDatabase, synthesis
from trainingData.inputLoading.AudioSample import AudioSample
from trainingData.trainingDatabase import TrainingDatabaseConnection

# Constants for the config key
from utils import _CHECKPOINT_PATH_KEY, perform_preprocessing

_BATCH_SIZE_KEY = "batch_size"
_LEARN_RATE_KEY = "learn_rate"
_ADJUST_LEARN_RATE_AFTER_EPOCHS = "adjust_learn_rate_after_epochs"
_ADJUST_LEARN_RATE_MULTIPLIER = "adjust_learn_rate_multiplier"
_WEIGHT_DECAY_KEY = "weight_decay"
_NUMBER_OF_EPOCHS_KEY = "number_of_epochs"
_TEST_TRAIN_DATA_RATIO_KEY = "test_train_data_ratio"
_TRAIN_TEST_SPLIT_SEED_KEY = "train_test_split_seed"
_CHECKPOINT_EPOCH_INTERVAL_KEY = "checkpoint_epoch_interval"
_BEST_MODEL_PATH_KEY = "best_model_path"


def train_model(
        model_name: str,
        config: PipelineConfigLoader,
        database: TrainingDatabaseConnection,
        samples: Optional[List[AudioSample]] = None,
        is_pre_split: bool = False,
        visualize=False,
        cpu=False
):
    """
    Called to start the training process.

    :param model_name The name of the model to train
    :param config: The configuration to use for the training.
    :param database: The training database to load samples from in case none a given.
    :param samples: The samples to use for training. If this is None they are loaded from the DB.
    :param is_pre_split: If the given samples are pre-split into test and train. If so the first
    element shall be the training set and the second the test set.
    :param visualize If true, some training data is shown
    :param cpu Indicates if the cpu should be used in training
    :return: None.
    """
    logging.info("=== START TRAINING ===")
    # Check if there are no samples given
    if not samples:
        # Load the samples from the database
        samples = trainingDatabase.load_samples_from_database(database)
        logging.info(f"Loaded {len(samples)} samples from the database.")
        if not samples:
            samples = synthesis.perform_synthesis(config, database)
            is_pre_split = True
    # Preprocess the samples
    if is_pre_split:
        data = [perform_preprocessing(config, sample_split, cpu=cpu) for sample_split in samples]
        # Get the samples
        quantized_samples = list(
            itertools.chain([sample_split[0] for sample_split in data])
        )
        # And labels into separate lists
        labels = list(itertools.chain([sample_split[1] for sample_split in data]))
        # Get the number of unique labels in the training set
        number_of_unique_labels = data[0][2]
        # Get the labels as a list for logging
        labels_list = torch.concat(labels).tolist()
        # The number of in features is the number of MFCCs
        number_in_features = quantized_samples[0].shape[1]
    else:
        quantized_samples, labels, number_of_unique_labels = perform_preprocessing(
            config, samples, visualize, cpu
        )
        # Get the labels as a list for logging
        labels_list = labels.tolist()
        # Get the number of in features
        number_in_features = len(quantized_samples[0])
    # Count the number of samples per category
    logging.info("Number of samples per category:")
    for label in set(labels_list):
        logging.info(
            f"{utils.get_category_name(config, label)}: {labels_list.count(label)}"
        )
    # Create the data loaders
    test_loader, train_loader = _prepare_data_loaders(
        config, labels, quantized_samples, is_pre_split
    )

    # be able to train multiple models in line, to avoid doing preprocessing multiple times
    for name in model_name.split(','):
        # Create and try to load the neural network
        network = NeuralNetwork(
            config=config,
            in_features=number_in_features,
            number_of_categories=number_of_unique_labels,
            model_name=name,
            cpu=cpu
        )

        # Log
        logging.info(
            f"Created network with {number_in_features} input features and {number_of_unique_labels} output categories."
        )
        try:
            # Train it according to the config
            network.train_network(
                train_loader,
                test_loader,
                config.train_config[_LEARN_RATE_KEY],
                config.train_config[_WEIGHT_DECAY_KEY],
                config.train_config[_NUMBER_OF_EPOCHS_KEY],
                pathlib.Path(config.train_config[_CHECKPOINT_PATH_KEY] + '/' + name),
                config.train_config[_BEST_MODEL_PATH_KEY],
                config.train_config[_CHECKPOINT_EPOCH_INTERVAL_KEY],
                config.train_config[_ADJUST_LEARN_RATE_AFTER_EPOCHS],
                config.train_config[_ADJUST_LEARN_RATE_MULTIPLIER],
            )
        except Exception:
            # if training failed, just continue with the next
            logging.info(f'Training of {name} failed')
            ex_type, ex, tb = sys.exc_info()
            traceback.print_tb(tb)
        # delete model to free gpu
        del network.model


def _prepare_data_loaders(config, labels, quantized_samples, is_pre_split):
    """
    Prepares the data loaders for the training.

    Performs the train/test split and creates the data loaders.
    :param config: The config to use.
    :param labels: The labels to use.
    :param quantized_samples: The input samples to use.
    :param is_pre_split: If this is true the samples are split based on index and not on random sampling
    :return: The test and train data loaders.
    """
    # Split the samples and labels into training and test data
    if not is_pre_split:
        # Calculate the length of the train and test data set according to the given ratio
        test_data_length = int(
            len(quantized_samples) * config.train_config[_TEST_TRAIN_DATA_RATIO_KEY]
        )
        train_data_length = len(quantized_samples) - test_data_length
        # Create the rng to use
        generator = torch.Generator()
        # Check if a manual seed shall be used
        if config.train_config[_TRAIN_TEST_SPLIT_SEED_KEY] != -1:
            generator.manual_seed(config.train_config[_TRAIN_TEST_SPLIT_SEED_KEY])
        # Use sampling
        train_set, test_set = torch.utils.data.random_split(
            list(zip(quantized_samples, labels)), [train_data_length, test_data_length]
        )
    else:
        # Use the pre-split, zip them together with their labels
        train_set = list(zip(quantized_samples[0], labels[0]))
        test_set = list(zip(quantized_samples[1], labels[1]))
    # Create the data loaders
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=config.train_config[_BATCH_SIZE_KEY], shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=config.train_config[_BATCH_SIZE_KEY], shuffle=True
    )
    return test_loader, train_loader
