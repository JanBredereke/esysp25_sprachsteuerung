import logging
import pathlib
from typing import Any, MutableMapping

import tomli as toml


class PipelineConfigLoader:
    """
    Loads the pipeline config file and returns the config as a dictionary
    """

    # The key used in the TOML-config to hold the config of the input data
    _INPUT_DATA_KEY = "input_data"
    # The key used for the synthesis config
    _SYNTHESIS_KEY = "synthesis"
    # The key used for the train config
    _TRAINING_KEY = "training"
    # The key used for inference config
    _INFERENCE_KEY = "inference"

    def __init__(self, config_file: pathlib.Path):
        """
        Creates the config loader with the given config file.
        :param config_file: The config file to use.
        """
        # Check if the config file exists
        if not config_file.exists():
            raise FileNotFoundError(f"Config file {config_file} does not exist.")
        # Save the config file path
        self._config_file = config_file
        # Init the config file
        self._config = None

    def load_config(self):
        """
        Loads the config.
        :return: None.
        """
        # Load the config file
        with self._config_file.open("rb") as config_file:
            self._config = toml.load(config_file)
        # Log config file path
        logging.info(f"Loaded config from {self._config_file}")

    def _load_config_if_not_loaded(self):
        """
        Helper used to load the config if it was not loaded before but is accessed.
        :return: None.
        """
        if self._config is None:
            self.load_config()

    @property
    def input_data_config(self) -> dict:
        """
        :return: Returns the configuration for the input data.
        """
        # Load the config if it was not loaded before
        self._load_config_if_not_loaded()
        # Access the input data config
        return self._config[PipelineConfigLoader._INPUT_DATA_KEY]

    @property
    def synthesis_config(self) -> dict:
        """
        :return: Returns the config for the synthesis.
        """
        # Load the config if it was not loaded before
        self._load_config_if_not_loaded()
        # Access the synthesis config
        return self._config[PipelineConfigLoader._SYNTHESIS_KEY]

    @property
    def train_config(self) -> dict:
        """
        :return: Returns the config for the training.
        """
        # Load the config if it was not loaded before
        self._load_config_if_not_loaded()
        # Access the training config
        return self._config[PipelineConfigLoader._TRAINING_KEY]

    @property
    def inference_config(self) -> dict:
        """
        :return: Returns the config for the inference.
        """
        # Load the config if it was not loaded before
        self._load_config_if_not_loaded()
        # Access the inference config
        return self._config[PipelineConfigLoader._INFERENCE_KEY]

    @property
    def db_path(self) -> str:
        """
        :return: Returns the path to the database.
        """
        # Load the config if it was not loaded before
        self._load_config_if_not_loaded()
        # Access the synthesis config
        return self._config[PipelineConfigLoader._SYNTHESIS_KEY]["db_path"]
