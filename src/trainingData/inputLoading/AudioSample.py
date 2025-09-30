from collections import namedtuple
from dataclasses import dataclass
from typing import List

import numpy as np


SampleCategoryMarker = namedtuple(
    "SampleCategoryMarker", ["start_index", "end_index", "name"]
)


@dataclass
class AudioSample:
    """
    Dataclass to hold the raw data and category of an audio sample.
    """

    data: np.ndarray
    # The sample rate in Hz for the sample
    sample_rate: int
    # Holds the categories together with their start and end indexes
    category: List[SampleCategoryMarker]

    @property
    def is_silence(self) -> bool:
        """
        Returns True if this is silence.
        :return: True if this is silence, False otherwise.
        """
        return not self.category and (self.data == 0).all()

    def __eq__(self, other):
        """
        Tests samples on equality.

        They are equal if the data, sample rate and categories are equal.
        :param other: The other to compare to.
        :return: True if equal, else False.
        """
        return np.array_equal(self.data, other.data) and self.sample_rate == other.sample_rate and self.category == other.category
