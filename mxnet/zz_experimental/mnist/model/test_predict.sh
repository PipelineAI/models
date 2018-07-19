conda env create -n scikit --file pipeline_conda_environment.yml
source activate scikit

PIPELINE_INPUT_PATH=../input/predict/test_request.json PIPELINE_RESOURCE_RUNTIME=python PIPELINE_RESOURCE_SUBTYPE=scikit PIPELINE_RESOURCE_NAME=linear PIPELINE_RESOURCE_TAG=a PIPELINE_RESOURCE_RUNTIME=python PIPELINE_RESOURCE_CHIP=cpu \
   python test.py
