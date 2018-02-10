#!/bin/bash

PIPELINE_MODEL_RUNTIME=python PIPELINE_MODEL_TYPE=pytorch PIPELINE_MODEL_NAME=mnist PIPELINE_MODEL_TAG=pytorchcpu python test_predict.py

