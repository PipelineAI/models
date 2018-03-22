#!/bin/bash

capture_tpu_profile \
        --tpu_name=$TPU_NAME \
	--logdir=$MODEL_DIR &

python resnet_main.py \
        --master=grpc://10.0.101.2:8470 \
	--data_dir=gs://cloud-tpu-test-datasets/fake_imagenet \
	--model_dir=gs://tpu-demo/rom/output2 \
	--use_tpu=True 
