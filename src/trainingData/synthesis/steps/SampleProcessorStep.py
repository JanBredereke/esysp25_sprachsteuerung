import itertools
import random
from typing import List

import utils
from trainingData.inputLoading.AudioSample import AudioSample
from trainingData.synthesis import SynthesisStep, SampleProcessor
import numpy as np

# Ensure all sample processors are loaded
from trainingData.synthesis.sampleProcessors import *

#


class SampleProcessorStep(SynthesisStep):
    """
    A synthesis step in which sample processors are applied to the sample(s).
    """

    _ALWAYS_APPLY = 1.0
    _NEVER_APPLY = 0.0

    def __init__(self, synthesis_config: dict):
        """
        Initialises the step by loading the sample processors.
        """
        # Init super
        super().__init__(synthesis_config)
        # Load the processors by loading all subclasses of SampleProcessor
        self._sample_processors = [
            processor(synthesis_config)
            for processor in utils.get_all_subclasses(SampleProcessor)
        ]
        # Update the sample processors with the step specific config
        for processor in self._sample_processors:
            processor.update_config_from_dict(self._step_config)
        # Save the ones that support silence
        self._silence_processors = [
            processor
            for processor in self._sample_processors
            if processor.is_applicable_to_empty_sample
        ]

    def _apply_processors(self, samples, synthesis_config):
        """
        Applies the given processors to the given samples according the minimum and maximum amount of processors
        to apply for the step and the probabilities of the processors.

        :param samples: The samples to apply the processors to.
        :param synthesis_config: The synthesis config.
        :return: The samples after applying the processors.
        """
        # Take out processors which have a probability of 100%
        processors_to_always_apply = [
            processor
            for processor in self._sample_processors
            if processor.apply_probability == SampleProcessorStep._ALWAYS_APPLY
        ]
        # Normalize the probabilities of the processors
        probabilities = np.array(
            [
                processor.apply_probability
                for processor in self._sample_processors
                if processor.apply_probability != SampleProcessorStep._ALWAYS_APPLY
            ]
        )
        probabilities /= np.sum(probabilities)
        # Get the number of processors to apply for this step
        min_number_processors = synthesis_config[self.config_key][
            "min_number_processors"
        ]
        max_number_processors = synthesis_config[self.config_key][
            "max_number_processors"
        ]
        # Clip min to the number of processors to always apply
        min_number_processors = max(
            min_number_processors, len(processors_to_always_apply)
        )
        # Clip the max to the number of processors
        max_number_processors = min(max_number_processors, len(self._sample_processors))
        # Iterate over the samples
        for sample in samples:
            self._apply_processors_to_sample(
                max_number_processors,
                min_number_processors,
                probabilities,
                processors_to_always_apply,
                sample,
            )
        # Return the samples
        return samples

    def _apply_processors_to_sample(
        self,
        max_number_processors,
        min_number_processors,
        probabilities,
        processors_to_always_apply,
        sample,
    ):
        """
        Applies the processors to the given sample.
        :param max_number_processors: The maximum number of processors to apply to the sample.
        :param min_number_processors: The minimum number of processors to apply to the sample.
        :param probabilities: The probabilities of the processors to apply.
        :param processors_to_always_apply: The processors to always apply. These are always used
        and not sampled from randomly.
        :param sample: The sample to apply the processors to.
        :return: None, this is performed in place.
        """
        # Check if there are random processors left to apply
        if min_number_processors < max_number_processors:
            # Get the number of processors to use for the sample
            number_random_processors = random.randint(
                min_number_processors, max_number_processors
            )
            # Get our processors
            processors = self._sample_processors
            # Subtract the number of processors to always apply
            number_random_processors -= len(processors_to_always_apply)
            # If this is silence
            if sample.is_silence:
                (
                    number_random_processors,
                    processors,
                    probabilities,
                ) = self._handle_silence(number_random_processors)
            # Choose the sample processors to use
            random_processors_to_use = self._choose_random_processors(
                number_random_processors,
                probabilities,
                processors,
                processors_to_always_apply,
            )
        else:
            random_processors_to_use = []
        # Perform the synthesis
        for processor in itertools.chain(
            processors_to_always_apply, random_processors_to_use
        ):
            sample = processor.process_sample(sample)

    @staticmethod
    def _choose_random_processors(
        number_random_processors, probabilities, processors, processors_to_always_apply
    ):
        """
        Chooses the random processors to use.
        :param number_random_processors: The number of processors to choose.
        :param probabilities: The probabilities for each processor.
        :param processors: The processors to choose from.
        :param processors_to_always_apply: The processors to always apply. They cannot be chosen.
        :return: The processors.
        """
        # Filter out the processors to always apply
        filtered_processors = [
            processor
            for processor in processors
            # Filter out the processors that shall always be used
            if processor not in processors_to_always_apply
        ]
        try:
            random_processors_to_use = (
                np.random.default_rng().choice(
                    # Filter out the processors that shall always be used, and the processors that shall never
                    # be used
                    [
                        processor
                        for i, processor in enumerate(filtered_processors)
                        # Filter out the processors that shall never be used
                        if probabilities[i] > SampleProcessorStep._NEVER_APPLY
                    ],
                    number_random_processors,
                    p=probabilities,
                )
                if number_random_processors > 0
                else []
            )
        except ValueError:
            # No processors to choose from
            random_processors_to_use = []
        return random_processors_to_use

    def _handle_silence(self, number_random_processors):
        """
        Handles choosing the processors in case the current sample is silence.
        :param number_random_processors: The number of random processors to choose.
        This is altered to be at least one, and at max the number of processors which support silence.
        :return: The number of random processors to use, the processors to sample from, and the probabilities.
        """
        # We at least apply one processor
        number_random_processors = max(1, number_random_processors)
        # But at max the number of processors which support silence
        number_random_processors = min(
            len(self._silence_processors),
            number_random_processors,
        )
        # Re-do the probabilities
        probs = [processor.apply_probability for processor in self._silence_processors]
        # Check if not all are 0
        if sum(probs) == 0:
            # If so, choose a random one to overwrite the probability of
            probs[np.random.randint(0, len(probs))] = 1
        # Normalize the probabilities
        probs = probs / np.sum(probs)
        # Switch to only silence processors
        return number_random_processors, self._silence_processors, probs

    def perform(self, samples: List[AudioSample]):
        """
        Applies the known sample processors, according
        - to the min and max number of processors to be applied according to the config and
        - the probabilities of the loaded sample processors.

        This uses the step specific config to determine the min and max number of processors to apply.
        :param samples: The sample(s) to apply the processors on.
        :return: The processed samples.
        """
        if not samples:
            return samples
        # Apply the processors and return the samples
        return self._apply_processors(samples, self._synthesis_config)
