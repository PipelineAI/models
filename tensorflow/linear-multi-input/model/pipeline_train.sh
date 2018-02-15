PIPELINE_MODEL_RUNTIME=tfserving \
PIPELINE_MODEL_TYPE=tensorflow \
PIPELINE_MODEL_NAME=linear \
PIPELINE_MODEL_TAG=multi \
PIPELINE_INPUT_PATH=../input/ \
PIPELINE_OUTPUT_PATH=./pipeline_tfserving/ \
  python pipeline_train.py
