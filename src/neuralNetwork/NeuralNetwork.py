import logging
import pathlib
from os.path import dirname
import numpy as np
import torch.optim as optim
import torch
from brevitas.onnx import export_finn_onnx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from torch.utils.tensorboard import SummaryWriter
from utils import get_all_subclasses, init_checkpointing
from neuralNetwork.Model import SuperModel, DefaultModel, SuperModelSeq
from pipelineConfig.PipelineConfigLoader import PipelineConfigLoader
from sklearn.metrics import confusion_matrix
import sys
import traceback
# you maybe need to install finn git repo with "git clone https://github.com/Xilinx/finn/" and copy the visualization
# file into the dist directory
from finn.util.visualization import showInNetron

# if cuda is available use GPU to train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _select_model(model_name, in_features, num_categories):
    for model in [*get_all_subclasses(SuperModel), *get_all_subclasses(SuperModelSeq)]:
        if model.__name__ == model_name:
            if issubclass(model, SuperModel):
                return model(in_features, num_categories)
            else:
                return model(in_features, num_categories).model
    logging.info("No model found, use DefaultModel")
    return DefaultModel(in_features, num_categories)


class NeuralNetwork:
    """
    Implements the neural network model for classification.
    """

    def __init__(self, config: PipelineConfigLoader, in_features=0, number_of_categories: int = 0,
                 model_name='DefaultModel', load_checkpoint=True, cpu=False):
        """
        Initialises the network.

        :param in_features: The number of features in the input data.
        :param number_of_categories: The number of categories to classify.
        :param model_name The name of the model to load. Default value is DefaultModel
        :param load_checkpoint Indicates, weather the checkpoint should be loaded. Initialises checkpointing as well
        :param cpu Force using cpu
        """
        super().__init__()
        # Save params
        self._in_features = in_features
        self._number_of_categories = number_of_categories
        self.config = config
        self.model_name = model_name
        # if cuda is available use GPU to train the model
        self.device = torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')
        # Create the layers
        if load_checkpoint:
            try:
                self._load_model(config, model_name)
            except ValueError:
                logging.info("Using initial state")
                self.model = _select_model(model_name.split('_')[0], in_features, number_of_categories)
        else:
            self.model = _select_model(model_name.split('_')[0], in_features, number_of_categories)

        # Load network into GPU or CPU as defined in device
        self.model.to(self.device)

    def _load_model(self, config, model_name):
        """
        Loads the model with its weights, if they exists

        :param config: The config to use.
        :param model_name The name of the model
        """
        # Load the checkpoint
        self.checkpoint = init_checkpointing(config, model_name)
        # Check if the checkpoint is valid
        if not self.checkpoint:
            # Would use a random network, inference would make no sense
            raise ValueError("No checkpoint found to load. See log.")
        # Get the network parameters from the checkpoint
        # The in features are encoded in the first layer as the shape of the weights
        self._in_features = self.checkpoint["in_features"]
        # Same for the output features (-2 since last is the number of batches the batch norm was trained on)
        self._number_of_categories = self.checkpoint["number_of_categories"]
        # Create the network
        self.model = _select_model(model_name.split('_')[0], self._in_features, self._number_of_categories)
        # Load the weights
        self.model.load_state_dict(self.checkpoint["model_state_dict"])

    def forward(self, x):
        """
        Implement the forward pass of the network.

        :param x: The input to the network.
        :return: The output of the network.
        """
        x = x.to(self.device)
        return self.model.forward(x)

    def train_network(
            self,
            train_set,
            test_set,
            lr,
            weight_decay,
            epochs,
            checkpoint_dir,
            best_model_path,
            checkpoint_interval,
            adjust_lr_interval,
            adjust_lr_factor,
    ):
        """
        Trains the network.

        :param train_set: The inputs to train on. Zipped together with the labels.
        :param test_set: The inputs to test on.
        :param lr: The learning rate.
        :param weight_decay: The weight decay.
        :param epochs: The number of epochs to train for.
        :param checkpoint_dir: The directory to save checkpoints to.
        :param best_model_path: The path to save the best model to.
        :param checkpoint_interval: The interval to save checkpoints at.
        :param adjust_lr_interval: The interval to adjust the learning rate at in epochs.
        :param adjust_lr_factor: The factor to adjust the learning rate by.
        :return: None.
        """
        # Create a summary writer for tensorboard
        writer = SummaryWriter(filename_suffix=self.model_name)
        # Create the loss function
        criterion = torch.nn.CrossEntropyLoss()
        # Create the optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        # Create the scheduler, use 1, (i.e. no adjustment) as the initial value for gamma
        adjust_lr_interval = max(1, adjust_lr_interval)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=adjust_lr_interval,
            gamma=adjust_lr_factor if adjust_lr_interval > 0 else 1,
        )
        # Check if we have a checkpoint
        if self.checkpoint:
            # Load the model
            self.model.load_state_dict(self.checkpoint["model_state_dict"])
            self.model.to(self.device)
            # Set the optimizer state
            optimizer.load_state_dict(self.checkpoint["optimizer_state_dict"])
            # Set the best accuracy
            best_accuracy = self.checkpoint.get("best_accuracy", 0)
        else:
            # Set the best accuracy to 0
            best_accuracy = 0
        # Set model and loss to train mode
        self.model.train()
        criterion.train()
        # Iterate over the epochs
        for epoch in range(epochs):
            # Train for one epoch
            epoch_loss = self._train_for_one_epoch(
                criterion.to(self.device), epoch, optimizer, train_set, writer
            )
            # Calculate the loss
            epoch_loss /= len(train_set)
            # Write the loss to the summary writer
            writer.add_scalar("Loss/Train", epoch_loss, epoch)
            # Step the scheduler
            scheduler.step()
            # Save the model if the number of epochs is a multiple of the checkpoint interval
            if checkpoint_interval == 0 or epoch % checkpoint_interval == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": epoch_loss,
                        "in_features": self._in_features,
                        "number_of_categories": self._number_of_categories,
                    },
                    checkpoint_dir / f"checkpoint_{epoch}.pth",
                )
                # Log
                logging.info(f"Saved checkpoint checkpoint_{epoch}.pth")
            with torch.no_grad():
                # Evaluate the model on the test set
                model_accuracy = self._perform_eval(test_set, criterion, writer, epoch)
                # Save the model if it is the best model
                if model_accuracy > best_accuracy:
                    best_accuracy = model_accuracy
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": epoch_loss,
                            "accuracy": best_accuracy,
                            "in_features": self._in_features,
                            "number_of_categories": self._number_of_categories,
                        },
                        checkpoint_dir / f"{best_model_path}.pth",
                    )
                    # Log
                    logging.info(
                        f"Saved best model {best_model_path}.pth with accuracy {best_accuracy}"
                    )
            # Log epoch finish
            logging.info(f"Epoch {epoch + 1}/{epochs} finished")

    def _train_for_one_epoch(self, criterion, epoch, optimizer, train_set, writer):
        """
        Iterates over the training set once and performs the training-

        :param criterion: The criterion to use.
        :param epoch: The current epoch ID.
        :param optimizer: The optimizer to use.
        :param train_set: The train set.
        :param writer: The tensorboard writer to use for logging.
        :return: The loss of the epoch.
        """
        # Sum up the losses for the epoch
        epoch_loss = 0
        # Iterate over the training set
        for i, (inputs, labels) in enumerate(train_set):
            labels = labels.to(self.device)
            # Convert the type of the train inputs and test set
            inputs = inputs.type(torch.float32).to(self.device)
            outputs = self.forward(inputs.to(self.device)).to(self.device)
            outputs = outputs.to(self.device)
            loss = criterion(outputs, labels).to(self.device)
            # Write the loss of the current batch to the tensorboard
            writer.add_scalar("Loss/Train per Batch", loss, epoch * len(train_set) + i)
            # Add the loss to the running loss
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return epoch_loss

    def _perform_eval(self, test_set, criterion, writer, epoch):
        """
        Evaluates the network based on the labels.
        :param test_set: The set to test on.
        :param criterion: The criterion to use for test set loss.
        :param writer: The tensorboard writer.
        :param epoch: The current epoch.
        :return: The accuracy of the network.
        """
        label_list = list(self.config.input_data_config)  # ['forward', 'back', 'left', 'right', 'stop']
        label_list.append('none')  # + 'none'
        num_labels = len(self.config.input_data_config) + 1  # 5+1

        # Set self and criterion to eval mode
        self.model.eval()
        criterion.eval()
        # The loss over the test set
        running_loss = 0
        running_accuracy = 0
        running_certainty = 0

        # Iterate over the test set
        for i, (inputs, labels) in enumerate(test_set):
            labels = labels.to(self.device)
            # Convert the type of the test inputs
            inputs = inputs.type(torch.float32).to(self.device)
            outputs = self.forward(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            outputs = torch.nn.Softmax(dim=1)(outputs)
            # Get the indexes of the max values of each output
            prediction_certainty, predicted = torch.max(outputs.data, 1)
            # Compare the predicted values with the labels and sum the number of matches
            correct_prediction_mask = predicted == labels

            # Create confusion matrix
            if epoch % 10 == 0:
                # Prepare comparison between actual and predicted command
                conf_predicted = predicted
                conf_labels = labels
                # conf_predicted = conf_predicted.to('cpu').numpy()
                # conf_labels = conf_labels.to('cpu').numpy()
                # copy to cpu because we need numpy
                y_actu = pd.Series(conf_labels.to('cpu').numpy(), name='Actual')
                y_pred = pd.Series(conf_predicted.to('cpu').numpy(), name='Predicted')

                # Creating the confusion matrix
                cf_matrix_sci = confusion_matrix(y_actu, y_pred)
                df_cm_sci = pd.DataFrame(cf_matrix_sci, index=[i for i in range(num_labels)],
                                         columns=[i for i in range(num_labels)])

                # Put labels to matrix
                df_cm_sci.index = label_list
                df_cm_sci.columns = label_list

                # Plot and add to tensorboard
                plt.figure(figsize=(12, 7))
                cfm_heatmap_sci = sn.heatmap(df_cm_sci, annot=True).get_figure()
                writer.add_figure("Confusion Matrix", cfm_heatmap_sci, epoch)

            number_correct = correct_prediction_mask.sum().item()
            # Get the number of outputs
            total = labels.size(0)
            # Calculate the accuracy
            accuracy = number_correct / total
            running_accuracy += accuracy
            # Get the certainty of the correct predictions and take the average
            if number_correct > 0:
                certainty = (
                        prediction_certainty[correct_prediction_mask].sum().item()
                        / number_correct
                )
            else:
                certainty = 0
            running_certainty += certainty
        # Get the test set loss by dividing by the number of test inputs
        test_loss = running_loss / len(test_set)
        # Get the test set accuracy by dividing by the number of test inputs
        test_accuracy = running_accuracy / len(test_set)
        # Get the test set certainty by dividing by the number of test inputs
        test_certainty = running_certainty / len(test_set)
        # Write the results
        writer.add_scalar("Loss/Test", test_loss, epoch)
        writer.add_scalar("Accuracy/Test", test_accuracy, epoch)
        writer.add_scalar("Certainty/Test", test_certainty, epoch)
        # Reset to train mode
        self.model.train()
        criterion.train()
        return test_accuracy

    def verify(self):
        try:
            # Magic numbers from experience :D. Needs more than 1 in first dimension and 1990 is the width of the
            # input (normally)
            inputs = torch.tensor(np.zeros((2, 1990), dtype=np.int8)).type(torch.float32)
            # forward input one time, to check, weather the structure of the net is ok
            self.forward(inputs)
            return True
        except:
            logging.info(f"{self.model_name} cannot be trained.")
            ex_type, ex, tb = sys.exc_info()
            traceback.print_tb(tb)
            return False

    def convert(self, path: pathlib.Path):
        # Check, if the path has an onnx file at its end. If not, append the filename with model name and onnx ending
        if not path.parts[len(path.parts) - 1].endswith('.onnx'):
            path = path.joinpath(self.model_name + '.onnx')

        parent = dirname(path)
        # only create parents of the path, and not a .onnx directory
        if not pathlib.Path(parent).exists():
            pathlib.Path(parent).mkdir(parents=True)
        # for converting, the model needs to be on cpu
        self.model.to('cpu')
        # activate evaluation mode
        self.model.eval()
        logging.info(f"convert {self.model_name}")
        export_finn_onnx(self.model, export_path=path, input_shape=(1, 1990))


def show_onnx(file):
    showInNetron(file, '127.0.0.1', 8080)
