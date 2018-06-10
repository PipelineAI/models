conda env create -n scikit --file pipeline_conda_environment.yml
source activate scikit

PIPELINE_MODEL_PATH=. python pipeline_train.py
