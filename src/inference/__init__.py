import logging
import pathlib
import time
from typing import Optional

import torch
import numpy as np
import utils
from inference.inferenceDataProviders.AudioFileLoader import AudioFileLoader
from inference.inferenceDataProviders.PyAudioMicrophone import PyAudioMicrophone
from pipelineConfig.PipelineConfigLoader import PipelineConfigLoader
from neuralNetwork.NeuralNetwork import NeuralNetwork
from pydub import AudioSegment

_SAMPLE_RATE_KEY = "sample_rate"
_RECORDING_LENGTH_KEY = "recording_length"


def filter_voice_command(data: np.ndarray, threshold: float = 0.05):
    """
    Set all values in the signal to zero that are not part of the actual speech (based on amplitude thresholding).
    :param data: The audio data as a 1D NumPy array
    :param threshold: Amplitude threshold (normalized between 0 and 1)
    :return: F signal with silence outside the active region
    """
    norm_data = data.astype(np.float32)
    norm_data /= np.max(np.abs(norm_data)) + 1e-9

    active_indices = np.where(np.abs(norm_data) > threshold)[0]

    if len(active_indices) == 0:
        return np.zeros_like(data)

    # taking 23 milliseconds (1000 samples, 44.1 kHz) for buffering recognized voice command
    start = max(0, active_indices[0] - 1000)
    end = min(len(data), active_indices[-1] + 1000)

    filtered_signal = np.zeros_like(data)
    filtered_signal[start:end] = data[start:end]
    return filtered_signal


def start_inference(
        model_name: str,
        config: PipelineConfigLoader, audio_file_path: Optional[pathlib.Path] = None,
        visualize=False,
        cpu=False,
):
    """
    Used to start inference.
    :param model_name The name of the model to test
    :param config: The config to use.
    :param audio_file_path: The path to the audio file to use. If this is None, the microphone will be used.
    :param visualize Param to visualize the incoming Data
    :param cpu indicates, if the inference should be on cpu
    :return: None
    """
    device = torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')
    network = _load_network(config, model_name)
    recording_length = config.inference_config[_RECORDING_LENGTH_KEY]

    if audio_file_path:
        data_provider = AudioFileLoader(recording_length, audio_file_path)
    else:
        try:
            data_provider = PyAudioMicrophone(recording_length)
        except ImportError:
            logging.error("The PyAudio library is not installed. Please install it to use the microphone.")
            return

    while data_provider.has_data:
        data = data_provider.get_next_audio_input()

        # filter the recorded audio
        data.data = filter_voice_command(data.data)
        # saving audio for testing purposes
        raw_audio = AudioSegment(
            data.data.tobytes(),
            frame_rate=44100,
            sample_width=2,
            channels=1
        )


        if data_provider.does_require_preprocessing:
            data, _, _ = utils.perform_preprocessing(config, [data], visualize, cpu)

        data.to(device)

        network_output = (network.forward(data.type(torch.float32))).to(device)
        network_output = (torch.nn.functional.softmax(network_output, dim=1)).to(device)

        certainty, prediction = torch.max(network_output.data, 1)
        second_certainty, second_prediction = torch.max(
            network_output.data[network_output.data != certainty], 0
        )

        print(
            f"The prediction is {utils.get_category_name(config, prediction.item())} with a certainty of {certainty.item():.2f}."
        )
        print(
            f"Second guess is {utils.get_category_name(config, second_prediction.item() + 1 if second_prediction.item() >= prediction.item() else 0)} with a certainty of {second_certainty:.2f}."
        )


def _load_network(config, model_name):
    network = NeuralNetwork(config=config, model_name=model_name)
    network.model.eval()
    return network


def start_conversion(config: PipelineConfigLoader, audio_file_path: Optional[pathlib.Path] = None, visualize=False):
    recording_length = config.inference_config[_RECORDING_LENGTH_KEY]

    if audio_file_path:
        data_provider = AudioFileLoader(recording_length, audio_file_path)
    else:
        logging.error("No file found!")
        exit()

    file_counter = 1

    while data_provider.has_data:
        data = data_provider.get_next_audio_input()
        data.data = filter_voice_command(data.data)  # auch bei Conversion anwenden

        if data_provider.does_require_preprocessing:
            data, _, _ = utils.perform_preprocessing(config, [data], visualize)

        data.to('cpu')
        f_name = "outputInt8_" + str(file_counter) + ".npy"
        np.save(f_name, data.to('cpu').numpy(), allow_pickle=False)
        file_counter += 1
