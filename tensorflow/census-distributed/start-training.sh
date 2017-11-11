# Observed Behavior:  
#   If PS dies, Master needs restart.
#   If Master dies, training continues (however no checkpoint hooks or PS health checks are being performed)
#   If 1 Worker dies, training continues on remaining Worker(s).
#   If Worker is restarted, Training continues on the Worker.
#   All PS, Master, and Worker must be started before training starts.

# Based on these posts...
#   https://medium.com/onfido-tech/higher-level-apis-in-tensorflow-67bfb602e6c0
#   https://github.com/GoogleCloudPlatform/cloudml-samples/tree/master/census/customestimator/trainer
#   https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10_estimator

TF_CONFIG='{"environment": "cloud", "cluster": {"master": ["localhost:2222"], "ps": ["localhost:2223"], "worker": ["localhost:2224","localhost:2225"]}, "task": {"type": "master", "index": 0} }' python pipeline_train.py --num-epochs=25 --train-files=data/adult.data.csv --eval-files=data/adult.test.csv --job-dir=output/ &

TF_CONFIG='{"environment": "cloud", "cluster": {"master": ["localhost:2222"], "ps": ["localhost:2223"], "worker": ["localhost:2224","localhost:2225"]}, "task": {"type": "ps", "index": 0} }' python pipeline_train.py --num-epochs=25 --train-files=data/adult.data.csv --eval-files=data/adult.test.csv --job-dir=output/ &

TF_CONFIG='{"environment": "cloud", "cluster": {"master": ["localhost:2222"], "ps": ["localhost:2223"], "worker": ["localhost:2224","localhost:2225"]}, "task": {"type": "worker", "index": 0} }' python pipeline_train.py --num-epochs=25 --train-files=data/adult.data.csv --eval-files=data/adult.test.csv --job-dir=output/ &

TF_CONFIG='{"environment": "cloud", "cluster": {"master": ["localhost:2222"], "ps": ["localhost:2223"], "worker": ["localhost:2224","localhost:2225"]}, "task": {"type": "worker", "index": 1} }' python pipeline_train.py --num-epochs=25 --train-files=data/adult.data.csv --eval-files=data/adult.test.csv --job-dir=output/ &
