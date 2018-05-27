PIPELINE_MODEL_RUNTIME=tfserving \
PIPELINE_MODEL_TYPE=tensorflow \
PIPELINE_MODEL_NAME=census \
PIPELINE_MODEL_TAG=cpu \
PIPELINE_INPUT_PATH=../input/ \
PIPELINE_OUTPUT_PATH=./pipeline_tfserving/ \
  python pipeline_train.py
