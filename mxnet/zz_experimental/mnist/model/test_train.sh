conda env create -n scikit --file pipeline_conda_environment.yml
source activate scikit

PIPELINE_RESOURCE_PATH=. python pipeline_train.py
