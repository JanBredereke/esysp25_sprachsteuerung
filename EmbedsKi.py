"""
Main file for EmbedsKi.

Enables starting
- training data synthesis
- training the neural network
- testing the neural network during inference with either a single audio snippet
- (or live audio input.)
"""
import argparse
import logging
import pathlib
import sys

try:
    import inference
except ModuleNotFoundError:
    # If this is not found src is not in the pythonpath
    # and we need to add it.
    sys.path.append(str(pathlib.Path(__file__).parent / 'src'))
    # Retry
    import inference
import training
from pipelineConfig.PipelineConfigLoader import PipelineConfigLoader
from trainingData import synthesis
from trainingData.trainingDatabase import TrainingDatabaseConnection
from neuralNetwork.NeuralNetwork import NeuralNetwork, show_onnx

_LOG_DIR = pathlib.Path("logs")


def _parse_args():
    """
    Parses the command line arguments.
    :return: The results of the command line args.
    """
    parser = argparse.ArgumentParser(description="Starts to train the model.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--synthesis", action="store_true", help="Synthesize the training data."
    )
    parser.add_argument("--train", action="store_true", help="Train the model.")
    parser.add_argument("--test", action="store_true", help="Test the model.")
    parser.add_argument("--convertToNpy", action="store_true", help="Convert an Audiofile to a .npy-File")
    parser.add_argument(
        "--audioFile",
        help="Test only argument. Loads a file for inference. If this is None, the microphone will be used",
        type=pathlib.Path,
        default=None,
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the test and synthesis data"
    )
    parser.add_argument(
        "--modelName",
        type=str,
        default='DefaultModel'
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        help="The path to the config file.",
        default="config/TrainingDataPipelineConfig.toml",
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Convert the model to finn onnx"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify, if model can be trained"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force training on cpu",
        default=False
    )
    parser.add_argument(
        "--showOnnx",
        action="store_true",
        help="Show the onnx model with Netron"
    )
    parser.add_argument(
        "--log",
        help="Sets the loglevel.",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
    )
    parser.add_argument(
        "--logfile",
        type=pathlib.Path,
        help="Sets the logfile. Logs are placed in the 'logs' directory",
        default="lastRunEmbedsKi.log",
    )
    return parser.parse_args()


def _init_logging(command_line_args):
    """
    Initialises the logger from the command line args.
    :param command_line_args: The command line args.
    :return: None.
    """
    # Check if the log's directory exists and create if not
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    # Set the log level and logfile
    logging.basicConfig(
        level=command_line_args.log.upper(),
        handlers=[
            logging.FileHandler(_LOG_DIR / command_line_args.logfile, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def main():
    """
    Main function.
    :return: None.
    """
    # Parse command line args
    command_line_args = _parse_args()
    # Initialise logging
    _init_logging(command_line_args)
    # Test is only allowed if it is the single argument
    if command_line_args.test and (
            command_line_args.synthesis or command_line_args.train
    ):
        # This is not allowed
        raise ValueError("Test is only allowed if it is the single argument.")
    # Load the config file.
    config_loader = PipelineConfigLoader(command_line_args.config)

    # get model name(s) from args
    model_names: str = command_line_args.modelName
    # verify, that model can be trained
    if command_line_args.verify:
        valid_models = list()
        for model in model_names.split(','):
            if NeuralNetwork(config_loader, in_features=1990, number_of_categories=6, model_name=model).verify():
                # if verified, add it to valid model names
                valid_models.append(model)
        # join model names to use it like always
        model_names = ','.join(valid_models)
    # Connect to the database if synthesis or training is required
    if command_line_args.synthesis or command_line_args.train:
        database = TrainingDatabaseConnection(config_loader)
        # Check if synthesis is requested.
        samples = None
        if command_line_args.synthesis:
            # Synthesize the training data.
            samples = synthesis.perform_synthesis(config_loader, database)
        # Check if training is requested.
        if command_line_args.train:
            # Train the model.
            # If synthesis ran, the samples are pre-split
            training.train_model(model_names, config_loader, database, samples,
                                 is_pre_split=samples is not None, visualize=command_line_args.visualize,
                                 cpu=command_line_args.cpu)
    # Check if testing is requested.
    if command_line_args.test:
        # Test
        inference.start_inference(model_names, config_loader, command_line_args.audioFile,
                                  command_line_args.visualize)

    if command_line_args.convertToNpy:
        # Convert Audiofile
        inference.start_conversion(config_loader, command_line_args.audioFile)                           
    # convert the model
    if command_line_args.convert:
        for name in model_names.split(','):
            NeuralNetwork(config=config_loader, model_name=name, cpu=command_line_args.cpu).convert(
                pathlib.Path('finnModel'))

    if command_line_args.showOnnx:
        show_onnx(str(pathlib.Path('finnModel').joinpath(command_line_args.modelName + '.onnx')))


if __name__ == "__main__":
    main()
