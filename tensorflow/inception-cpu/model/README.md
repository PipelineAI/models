Derived from the following:
* https://github.com/tensorflow/serving/issues/354
* https://github.com/tensorflow/serving/tree/master/tensorflow_serving/example
* https://www.tensorflow.org/serving/serving_inception
* https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/inception_client.py
* https://github.com/faas-and-furious/inception-function

Training
```
$ curl -O http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz
$ tar xzf inception-v3-2016-03-01.tar.gz
```

Serving
```
$ cd pipeline_tfserving && tar xzf 0.tar.gz
```
