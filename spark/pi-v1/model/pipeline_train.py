from __future__ import print_function

import sys
from random import random
from operator import add
import math
from pyspark.sql import SparkSession
import mlflow

import calculate_pi

#user_id = <YOUR_USER_ID>
user_id = ''
model_name = 'sparkpi'
model_tag = 'v1'

if __name__ == "__main__":
    """
        Usage: pi [partitions]
    """
    tracking_uri = 'https://community.cloud.pipeline.ai'

    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = '%s%s-%s' % (user_id, model_name, model_tag)
    
    # This will create and set the experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        spark = SparkSession\
            .builder\
            .appName("PythonSparkPi")\
            .getOrCreate()

        partitions = 2
        n = 100000 * partitions

        mlflow.log_param('partitions', str(partitions))
        mlflow.log_param('n', str(n))
                
        def f(_):
            x = random() * 2 - 1
            y = random() * 2 - 1
            return 1 if x ** 2 + y ** 2 <= 1 else 0

        count = spark.sparkContext.parallelize(range(1, n + 1), partitions).map(f).reduce(add)
    
        calculated_pi = calculate_pi.calculate(count=count, n=n)
    
        print("Pi is roughly %f" % calculated_pi)
    
        error = abs(math.pi - calculated_pi)        
        mlflow.log_metric('error_pct', error * 100)

    spark.stop()

