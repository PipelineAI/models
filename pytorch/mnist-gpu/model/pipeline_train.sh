PIPELINE_MODEL_RUNTIME=python \
PIPELINE_MODEL_TYPE=pytorch \
PIPELINE_MODEL_NAME=mnist \
PIPELINE_MODEL_TAG=gpu \
PIPELINE_INPUT_PATH=../input/ \
PIPELINE_OUTPUT_PATH=./pipeline_tfserving/ \
  python pipeline_train.py
