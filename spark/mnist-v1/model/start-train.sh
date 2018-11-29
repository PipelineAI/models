#!/bin/bash

rm /tmp/pipeline_bundle.zip

sbt "runMain pipeline_train" 
