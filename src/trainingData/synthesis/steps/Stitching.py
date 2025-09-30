import random
from typing import List, Tuple

import numpy as np

from trainingData.inputLoading.AudioSample import AudioSample, SampleCategoryMarker
from trainingData.synthesis import SynthesisStep


class Stitching(SynthesisStep):
    """
    Stitches the samples together into one large sample.
    """

    @staticmethod
    def _choose_random_pause_lengths(total_pause_length, number_pauses):
        """
        Chooses number splits random intervals over total_pause_length.
        :param total_pause_length: The length of the pause to split up.
        Note that the length of the pauses do _not_ sum up to this, when there is a pause at the end (i.e. after the last
        sample, we place pauses in front of samples). The length of that pause is total_pause_length - sum(pause_lengths).
        :param number_pauses: The number of pauses to create.
        :return: The lengths of the pause intervals.
        """
        # Get the start indexes, and sort them
        start_indexes = sorted(
            np.random.randint(0, high=total_pause_length, size=number_pauses)
        )
        # Add the length as the last index
        start_indexes.append(total_pause_length)
        # Calculate the lengths between them
        return [
            start_indexes[i + 1] - start_indexes[i]
            for i in range(len(start_indexes) - 1)
        ]

    @staticmethod
    def _merge_sample_and_pause_positions(
        sum_pause_lengths, pause_lengths, sample_positions, sample_lengths
    ):
        """
        Merges the given samples in the given order with the given length of pauses in between.
        :param sum_pause_lengths: The sum of the lengths of the pauses.
        :param pause_lengths: The lengths of the pauses ordered as they shall be inserted. (A pause
        is inserted first, then a sample is placed after).
        :param sample_positions: The order of the indexes in the samples list in which to place the samples in.
        :param sample_lengths: The lenghts of the samples to place.
        :return: The stitched sample.
        """
        # Calculate the length of pauses in the end sample (i.e. where no samples are placed)
        # The end-index of the last placement
        last_end_index = 0
        # The start and end indexes
        indexes = [(0, 0)] * len(sample_lengths)
        # Choose a random length of pause, then insert a sample until all are placed
        for sample_position, current_pause_length in zip(
            sample_positions, pause_lengths
        ):
            # Get the current sample length
            sample_length = sample_lengths[sample_position]
            # Place the sample after the pause
            start_index = last_end_index + current_pause_length
            end_index = start_index + sample_length
            # Add the indexes to the list
            indexes[sample_position] = (start_index, end_index)
            # Update the last end index
            last_end_index = end_index
            # Update the pause length
            sum_pause_lengths = sum_pause_lengths - current_pause_length
        return indexes

    def _choose_stitch_indexes(
        self, sample_lengths, length_of_stitched_sample
    ) -> List[Tuple[int, int]]:
        """
        Prepares the stitching of the samples by choosing the start and end indexes.
        :param sample_lengths: The lengths of the samples.
        :param length_of_stitched_sample: The length of the stitched sample.
        :return: A list of tuples containing the start and end indexes.
        """
        # Nothing to stitch
        if len(sample_lengths) < 1:
            return []
        # Sum up the length of the given samples
        sum_sample_lengths = sum(sample_lengths)
        # Check that the sum of the lengths of the samples is less than the max length
        if sum_sample_lengths > length_of_stitched_sample:
            # Cannot stitch
            raise ValueError(
                "Cannot stitch together samples, too long, increase max length."
            )
        # Choose a random order for the samples
        sample_positions = random.sample(
            range(len(sample_lengths)), k=len(sample_lengths)
        )
        # Calculate the length of pauses in the stitched sample
        sum_pause_lengths = length_of_stitched_sample - sum_sample_lengths
        # Choose random pause lengths
        pause_lengths = self._choose_random_pause_lengths(
            sum_pause_lengths, len(sample_lengths)
        )
        # Merge the pauses and samples
        return self._merge_sample_and_pause_positions(
            sum_pause_lengths, pause_lengths, sample_positions, sample_lengths
        )

    def _stitch_together_samples(
        self, samples: List[AudioSample], max_length: int, sample_rate_hz: int
    ) -> List[AudioSample]:
        """
        Stitches together the samples into an audio clip of the given maxlength.
        :param samples: The samples to place.
        :param max_length: The max length of the samples in seconds.
        :return: The stitched together sample.
        """
        # Init the sample
        stitched_sample = np.zeros(max_length * sample_rate_hz, dtype=np.int16)
        # Choose the indexes for the samples to be placed
        indexes = self._choose_stitch_indexes(
            [len(sample.data) for sample in samples], len(stitched_sample)
        )
        # The indexes where the samples are placed
        end_sample_indexes = []
        # Iterate for each sample
        for sample, (start_index, end_index) in zip(samples, indexes):
            # Get the audio
            sample_audio = sample.data
            # Add the sample to the stitched sample
            stitched_sample[start_index:end_index] = sample_audio
            # Get the category of the sample
            category_name = sample.category[0].name
            # Add the indexes to the list
            end_sample_indexes.append(
                SampleCategoryMarker(start_index, end_index, category_name)
            )
        # Return the stitched sample and the indexes
        return [AudioSample(stitched_sample, sample_rate_hz, end_sample_indexes)]

    def perform(self, samples: List[AudioSample]):
        """
        Performs the stitching.
        :param samples: The samples to stitch.
        :return: A list containing one stitched sample.
        """
        return self._stitch_together_samples(
            samples,
            self._synthesis_config["max_length_seconds"],
            self._synthesis_config["sample_rate_hz"],
        )
