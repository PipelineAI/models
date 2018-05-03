Derived from https://github.com/tensorflow/models

As of this writing, there is a weird packaging structure inside that repo that we work around by moving the training python file, mnist.py, up a directory.

We're also renaming mnist.py to pipeline_train.py to conform to PipelineAI platform convention.
