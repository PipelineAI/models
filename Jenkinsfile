pipeline {
  agent any
  stages {
    stage('install cli-pipeline') {
      steps {
        sh 'pip install cli-pipeline==1.4.31 --ignore-installed --no-cache -U'
      }
    }
    stage('build train-census') {
      steps {
        sh 'pipeline train-server-build --model-type=tensorflow --model-name=census --model-tag=v1 --model-path=./tensorflow/census/model'
      }
    }
    stage('start train-census') {
      steps {
        sh '''pipeline train-server-start --model-type=tensorflow --model-name=census --model-tag=v1 --input-path=./tensorflow/census/input --output-path=./tensorflow/census/output --train-args="--train-files=training/adult.training.csv\\ --eval-files=validation/adult.validation.csv\\ --num-epochs=2\\ --learning-rate=0.025"
'''
      }
    }
    stage('logs train-census') {
      steps {
        sh '''pipeline train-server-logs --model-type=tensorflow --model-name=census --model-tag=v1
'''
      }
    }
    stage('sleep train-census') {
      steps {
        sleep 60
      }
    }
    stage('stop train-census') {
      steps {
        sh '''pipeline train-server-stop --model-type=tensorflow --model-name=census --model-tag=v1
'''
      }
    }
  }
}