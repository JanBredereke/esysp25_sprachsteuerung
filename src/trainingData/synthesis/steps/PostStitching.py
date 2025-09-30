from trainingData.synthesis.steps.SampleProcessorStep import SampleProcessorStep


class PostStitching(SampleProcessorStep):
    """
    Step in the synthesis of the training data which applies the sample processors to the stitched sample,
    i.e. the whole data.
    """

    # All handled in the superclass, we just need this for the config context
    pass
