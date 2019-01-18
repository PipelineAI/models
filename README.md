![PipelineAI Logo](https://pipeline.ai/assets/img/logo/pipelineai-logo.png)

# PipelineAI Quick Start (CPU + GPU)
Train and Deploy your ML and AI Models in the Following Environments:
* [Hosted Community Edition](https://quickstart.pipeline.ai/community)
* [Docker](https://quickstart.pipeline.ai/docker)
* [Kubernetes](https://quickstart.pipeline.ai/kubernetes)
* [AWS SageMaker](https://quickstart.pipeline.ai/sagemaker)

# Having Issues?  Contact Us Anytime... We're Always Awake.
* Slack:  https://joinslack.pipeline.ai
* Email:  [help@pipeline.ai](mailto:help@pipeline.ai)
* Web:  https://support.pipeline.ai
* YouTube:  https://youtube.pipeline.ai
* Slideshare:  https://slideshare.pipeline.ai
* Workshop:  https://workshop.pipeline.ai
* Meetup:  https://meetup.pipeline.ai
* Webinar:  https://webinar.pipeline.ai
* [Troubleshooting Guide](/docs/troubleshooting)

# Installing MLflow
Install mlflow via `pip install mlflow`

# Requirements
Python 3.6 (tensorflow is currently unsupported by Python 3.7)

Install tensorflow via `pip install tensorflow`

Install keras via `pip install keras`

Install PIL via `pip install pillow`

# Running Example
`mlflow run example/flower_classifier --no-conda`

# Launching UI
The MLflow Tracking UI will run at `<http://localhost:5000>`_.

`mlflow ui`

**Note** `mlflow ui` will not run from within the cloned repo.
