from trainingData.inputLoading.audioLoaders.AudioLoader import AudioLoader

# Ensure all loaders are imported
from trainingData.inputLoading.audioLoaders import *
from utils import get_all_subclasses


class AudioLoaderFactory:
    """
    Factory for loading audio files.
    """

    def __init__(self):
        pass

    @staticmethod
    def create(file_type):
        """
        Creates an AudioLoader object based on the file type.
        :param file_type: The file type of the audio file.
        :return: An AudioLoader object.
        """
        # Get the loader from all known loaders
        for loader in get_all_subclasses(AudioLoader):
            # Check if this loader is able to load the file type
            if loader.get_loader_extension() == file_type:
                # Return the loader
                return loader()
        # Did not return, we do not know a loader for this type
        raise Exception("No loader for file type: " + file_type)
