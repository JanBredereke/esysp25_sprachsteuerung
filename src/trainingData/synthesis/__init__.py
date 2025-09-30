import copy
import itertools
import logging
import random
from typing import List, Tuple

import numpy as np

import utils
from pipelineConfig.PipelineConfigLoader import PipelineConfigLoader
from trainingData import trainingDatabase
from trainingData.inputLoading.AudioSample import AudioSample
from trainingData.inputLoading.WordClassSampleLoader import WordClassSampleLoader
from trainingData.synthesis.steps.SynthesisStep import SynthesisStep
from trainingData.synthesis.sampleProcessors.SampleProcessor import SampleProcessor
from trainingData.synthesis.steps.SynthesisStepPipeline import SynthesisStepPipeline
from trainingData.trainingDatabase import TrainingDatabaseConnection


_EXPORT_CONFIG_KEY = "synthesisDataSave"
_TEST_TRAIN_KEY = "test_train_data_ratio"
_SAVE_SYNTHESIS_DATA_IN_DB_KEY = "save_synthesis_data_in_db"
_DO_EXPORT_SYNTHESIS_DATA_KEY = "do_export_synthesis_data"
_MAX_NUMBER_OF_WORDS_KEY = "max_number_of_words"
_MIN_NUMBER_OF_WORDS_KEY = "min_number_of_words"
_SAMPLE_RATE_HZ_KEY = "sample_rate_hz"
_LENGTH_SECONDS_KEY = "max_length_seconds"
_NUMBER_OUTPUT_SAMPLES_KEY = "number_output_samples"


def perform_synthesis(
    config_loader: PipelineConfigLoader, database: TrainingDatabaseConnection
) -> Tuple[List[AudioSample], List[AudioSample]]:
    """
    Performs the data synthesis.
    :param config_loader: The previously loaded config.
    :param database: The database to use for saving the samples if required.
    :return: The samples split into train and test.
    """
    logging.info("=== START SYNTHESIS ===")
    # Create the input loader
    input_loader = WordClassSampleLoader(config_loader.input_data_config)
    # Load the samples
    loaded_samples = input_loader.load_configured_categories()
    # Get the number of categories
    number_of_categories = len(loaded_samples)
    # Split each category into train and test
    split_categories = [_perform_test_train_split(samples, config_loader.train_config[_TEST_TRAIN_KEY]) for samples in loaded_samples.values()]
    # Pack train and test into separate lists
    test_set = [samples for category_samples in split_categories for samples in category_samples[1]]
    train_set = [samples for category_samples in split_categories for samples in category_samples[0]]
    # Log the number of loaded samples
    logging.info(f"Loaded {len(test_set) + len(train_set)} input samples for synthesis.")
    # List for the output
    synthesised_samples_test = []
    synthesised_samples_train = []
    # Create the pipeline
    pipeline = SynthesisStepPipeline(config_loader.synthesis_config)
    # Get the number of samples to produce
    number_of_samples = config_loader.synthesis_config[_NUMBER_OUTPUT_SAMPLES_KEY]
    # Perform the split for the number of samples
    number_of_test_samples = int(number_of_samples * config_loader.train_config[_TEST_TRAIN_KEY])
    # Log the number of stitched samples
    logging.info(f"Stitching {number_of_samples} samples together")
    # Iterate for as many samples as required
    for i in range(number_of_samples):
        # Log progress in percentage every 10% (or every sample if less than 10 samples)
        if i % (max(number_of_samples // 10, 1)) == 0:
            logging.info(f"Progress of synthesis: {(i / number_of_samples * 100):.2f}%")
        # Stitch and add the sample to the output samples
        if i < number_of_test_samples:
            # First stitch the test samples
            synthesised_samples_test.extend(_stitch_single_sample(test_set, config_loader.synthesis_config, pipeline, number_of_categories))
        else:
            # Then stitch the train samples
            synthesised_samples_train.extend(_stitch_single_sample(train_set, config_loader.synthesis_config, pipeline, number_of_categories))
    # Log that synthesis is complete
    logging.info("Synthesis complete")
    _do_synthesis_export(config_loader, database, synthesised_samples_test, synthesised_samples_train)
    # Return the samples
    return synthesised_samples_train, synthesised_samples_test


def _do_synthesis_export(config_loader, database, synthesised_samples_test, synthesised_samples_train):
    """
    Performs the export of the synthesised samples.
    :param config_loader: The config loader to use to read the config.
    :param database: The database to save the samples in if this is required.
    :param synthesised_samples_test: The test set.
    :param synthesised_samples_train: The train set.
    :return: None.
    """
    # Save the samples if configured
    if config_loader.synthesis_config.get(_EXPORT_CONFIG_KEY, None):
        # Get the export config
        export_config = config_loader.synthesis_config[_EXPORT_CONFIG_KEY]
        # Get all samples as a list
        all_samples = list(itertools.chain(synthesised_samples_train, synthesised_samples_test))
        # Log the save
        logging.info("Saving Synthesis Data")
        if export_config[_SAVE_SYNTHESIS_DATA_IN_DB_KEY]:
            # Save the samples to the DB
            trainingDatabase.save_samples_to_database(all_samples, database)
        # Export the samples if configured
        if export_config[_DO_EXPORT_SYNTHESIS_DATA_KEY]:
            # Export the samples according to the config
            utils.export_samples_as_mp3(all_samples, export_config)


def _perform_test_train_split(samples: List[AudioSample], test_train_split) -> Tuple[List[AudioSample], List[AudioSample]]:
    """
    Splits the given samples into train and test.

    :param samples: The samples to split.
    :return: The split samples.
    """
    # Get the number of items in the test set
    number_test = int(len(samples) * test_train_split)
    # Get the test set
    test_set = random.sample(samples, number_test)
    # Get the remaining samples
    train_set = [sample for sample in samples if sample not in test_set]
    # Return the split
    return train_set, test_set


def _stitch_single_sample(samples, synthesis_config, pipeline, number_of_categories):
    """
    Stitches a single sample together by running through the SynthesisSteps.

    :param samples: The samples to choose from.
    :param synthesis_config: The config for the synthesis.
    :param pipeline: The synthesis pipeline to use.
    :param number_of_categories: The number of categories.
    :return: The stitched sample.
    """
    min_number_of_words = synthesis_config[_MIN_NUMBER_OF_WORDS_KEY]
    max_number_of_words = synthesis_config[_MAX_NUMBER_OF_WORDS_KEY]
    if min_number_of_words < 1:
        # Get the probability for each category
        one_category_prob = 1 / (number_of_categories + 1)
        # Sum together the probabilities for the non-none categories and split them over the max number of words
        category_prob = (one_category_prob * number_of_categories) / max_number_of_words
        # We can have none, normalize the probability distribution
        number_of_words = int(
            np.random.choice(
                np.linspace(
                    min_number_of_words,
                    max_number_of_words,
                    max_number_of_words - min_number_of_words + 1,
                ),
                p=[one_category_prob] + [category_prob] * max_number_of_words,
            )
        )
    else:
        # Sample a number of words to use
        number_of_words = random.randint(
            min_number_of_words,
            max_number_of_words,
        )
    # Select the samples to use
    current_samples = _select_samples(number_of_words, samples, synthesis_config)
    # Iterate over the pipeline
    for step in pipeline:
        # Perform the step
        current_samples = step.perform(current_samples)
    # Return the sample
    return current_samples


def _select_samples(number_of_words, samples, synthesis_config):
    """
    Selects a number of samples to use.

    Guarantees that the sum of the length of the samples is less than the max length.
    :param number_of_words: The number of samples to select.
    :param samples: The samples to select from.
    :param synthesis_config: The config for the synthesis.
    :return: The samples.
    """
    while True:
        # Choose one sample and perform a copy of it
        samples = (
            [
                copy.deepcopy(sample)
                for sample in random.choices(samples, k=number_of_words)
            ]
            if number_of_words > 0
            else []
        )
        # Check the length of the samples
        if (
            not sum(len(sample.data) for sample in samples)
            > synthesis_config[_LENGTH_SECONDS_KEY]
            * synthesis_config[_SAMPLE_RATE_HZ_KEY]
        ):
            # Found a valid sample, break
            return samples
