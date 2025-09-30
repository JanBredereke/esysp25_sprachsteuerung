import logging
import pathlib

from trainingData.inputLoading.SampleLoader import SampleLoader
from typing import Dict, List, Optional

from trainingData.inputLoading.AudioSample import AudioSample, SampleCategoryMarker


class WordClassSampleLoader(SampleLoader):
    """
    SampleLoader used for loading the input words together with their class labels.

    Loads the raw data through an AudioLoader delegate and marks the loaded audio sample with
    the corresponding input class (i.e. which word this is).
    """

    def __init__(self, input_config: dict):
        """
        Creates the loader.
        :param input_config: The config to use.
        """
        # Call super
        super().__init__()
        # Save the config
        self._input_config = input_config

    def load_configured_categories(
        self, filter_categories: Optional[List[str]] = None
    ) -> Dict[str, List[AudioSample]]:
        """
        Loads all inputs configured in the config of self.

        :param filter_categories: Used to filter out samples of certain categories.
        :return: The loaded inputs.
        """
        # If there are no categories to filter, init as an empty list
        if filter_categories is None:
            filter_categories = []
        # Dict to hold the samples sorted by category
        samples_per_category = {}
        # Iterate over the configured folders
        for _, input_folder in self._input_config.items():
            # Get the path
            input_path = pathlib.Path(input_folder["path"])
            # Check if the path exists
            if not input_path.exists():
                raise FileNotFoundError(
                    f"The input path '{input_path}' does not exist."
                )
            # Get the category
            category = input_folder["category"]
            # Check if the category shall be filtered
            if category in filter_categories:
                # Do not load this folder
                logging.info(f"Skipping input folder '{input_path}'.")
                continue
            # Check if the category is already known
            if category not in samples_per_category:
                # Add an emtpy list
                samples_per_category[category] = []
                # Log the new category
                logging.info(f"Found new input sample category: '{category}'.")
            # Iterate over the samples in the path
            for sample_path in input_path.iterdir():
                # Load the sample
                sample, sample_rate = super().load_sample_from_path(sample_path)
                # Create the sample with the category and add it to the list
                samples_per_category[category].append(
                    AudioSample(
                        sample,
                        sample_rate,
                        [
                            SampleCategoryMarker(
                                start_index=0, end_index=len(sample), name=category
                            )
                        ],
                    )
                )
        # Return the samples
        return samples_per_category
