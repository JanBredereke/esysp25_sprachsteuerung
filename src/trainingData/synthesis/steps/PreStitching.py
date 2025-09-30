from trainingData.synthesis.steps.SampleProcessorStep import SampleProcessorStep


class PreStitching(SampleProcessorStep):
    """
    Step in the synthesis of the training data which applies the sample processors to the non-stitched samples,
    i.e. the raw data.
    """

    # All handled in the superclass, we just need this for the config context
    pass
