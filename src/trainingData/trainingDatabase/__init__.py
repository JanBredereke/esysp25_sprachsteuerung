from typing import Iterable, Optional
import logging
from trainingData.inputLoading.AudioSample import AudioSample, SampleCategoryMarker
from trainingData.trainingDatabase.TrainingDatabaseConnection import (
    TrainingDatabaseConnection,
)
from trainingData.trainingDatabase.tables.TrainingData import TrainingData
from trainingData.trainingDatabase.tables.TrainingLabel import TrainingLabel


def save_samples_to_database(
    samples: Iterable[AudioSample], database_connection: TrainingDatabaseConnection
):
    """
    Saves the given samples to the database.
    :param samples: The samples to save.
    :param database_connection: The database connection to use.
    :return: None.
    """
    # Get a session
    session = database_connection.session
    # Iterate over the samples
    for sample in samples:
        # Get the length of the sample in seconds
        length_in_seconds = len(sample.data) / sample.sample_rate
        # Save the sample to the database
        database_object = TrainingData(
            sample_length_seconds=length_in_seconds,
            sample_rate_hz=sample.sample_rate,
            data=sample.data,
        )
        # Iterate the labels
        for label in sample.category:
            # Add the label to the database object
            database_object.training_labels.append(
                TrainingLabel(
                    start_index=label.start_index,
                    end_index=label.end_index,
                    label=label.name,
                )
            )
        # Add the database object to the session
        session.add(database_object)
    # Commit the session
    session.commit()
    # Close the session
    session.close()


def load_samples_from_database(
    database_connection: TrainingDatabaseConnection,
    number_samples: Optional[int] = None,
):
    """
    Loads samples from the given database.
    :param database_connection: The database to load from.
    :param number_samples: The number of samples to load, if None all are loaded from the database.
    :return: The loaded samples.
    """
    # Get a session
    session = database_connection.session
    # Build the query
    query = session.query(TrainingData)
    # If a number of samples is given
    if number_samples is not None:
        # Limit
        query = query.limit(number_samples)
    # Get the samples
    samples = query.all()
    num_samples = len(samples)
    # List for the loaded samples
    loaded_samples = []
    # Iterate the samples
    for i, sample in enumerate(samples):
        if i % (max(num_samples // 200, 1)) == 0:
            logging.info(f"Loading audio files: {(i / num_samples * 100):.2f}%...")
        audio_sample = _load_database_sample(sample)
        # Add the sample to the list
        loaded_samples.append(audio_sample)
    # Close the session
    session.close()
    # Return the loaded samples
    return loaded_samples


def _load_database_sample(sample):
    """
    Loads and returns a single sample.
    :param sample: The database object of the sample.
    :return: The loaded sample.
    """
    # Create a new audio sample
    audio_sample = AudioSample(
        sample_rate=sample.sample_rate_hz,
        data=sample.data,
        category=[
            SampleCategoryMarker(
                start_index=label.start_index,
                end_index=label.end_index,
                name=label.label,
            )
            for label in sample.training_labels
        ],
    )
    return audio_sample
