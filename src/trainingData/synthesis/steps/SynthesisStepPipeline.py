from trainingData.synthesis.steps.PostStitching import PostStitching
from trainingData.synthesis.steps.PreStitching import PreStitching
from trainingData.synthesis.steps.Stitching import Stitching


class SynthesisStepPipeline:
    """
    Holds the synthesis steps in the order they shall be processed in.
    """

    def __init__(self, synthesis_config: dict):
        """
        Creates the pipeline.
        """
        self._pipeline = [
            PreStitching(synthesis_config),
            Stitching(synthesis_config),
            PostStitching(synthesis_config),
        ]

    def __iter__(self):
        """
        Used to iterate over the pipeline.
        :return: The iterator over the pipeline.
        """
        return iter(self._pipeline)
