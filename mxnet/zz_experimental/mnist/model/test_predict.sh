conda env create -n scikit --file pipeline_conda_environment.yml
source activate scikit

PIPELINE_INPUT_PATH=../input/predict/test_request.json PIPELINE_MODEL_RUNTIME=python PIPELINE_MODEL_TYPE=scikit PIPELINE_MODEL_NAME=linear PIPELINE_MODEL_TAG=a PIPELINE_MODEL_RUNTIME=python PIPELINE_MODEL_CHIP=cpu \
   python test.py
