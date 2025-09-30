from sqlalchemy import Column, Integer, ForeignKey, String
from sqlalchemy.orm import relationship

from trainingData.trainingDatabase.TrainingDatabaseConnection import (
    TrainingDatabaseConnection,
)


class TrainingLabel(TrainingDatabaseConnection.Base):
    """
    Used to label training data.

    References one training sample together with the start and end-index within the samples data.
    """

    __tablename__ = "training_labels"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # The sample this applies to
    sample_id = Column(
        Integer, ForeignKey("training_data.id", ondelete="CASCADE"), nullable=False
    )
    sample = relationship("TrainingData", back_populates="training_labels")
    # The start index of the label
    start_index = Column(Integer, nullable=False)
    # The end index of the label
    end_index = Column(Integer, nullable=False)
    # The label
    label = Column(String, nullable=False)
