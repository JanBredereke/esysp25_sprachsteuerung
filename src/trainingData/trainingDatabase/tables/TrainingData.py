from sqlalchemy import Column, Integer, Float, PickleType
from sqlalchemy.orm import relationship

from trainingData.trainingDatabase.TrainingDatabaseConnection import (
    TrainingDatabaseConnection,
)


class TrainingData(TrainingDatabaseConnection.Base):
    """
    Table holding general information about the training data.
    """

    __tablename__ = "training_data"

    # The ID of the training data.
    id = Column(Integer, primary_key=True, autoincrement=True)
    # The length of the sample in seconds.
    sample_length_seconds = Column(Float)
    # The sample rate of the sample in Hz.
    sample_rate_hz = Column(Integer)
    # The corresponding training data, these are numpy arrays
    data = Column(PickleType)
    # The corresponding training labels
    training_labels = relationship(
        "TrainingLabel", back_populates="sample", order_by="TrainingLabel.start_index"
    )
