import time
  
import mleap.pyspark
from mleap.pyspark.spark_support import SimpleSparkSerializer

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

# refer to this post for more details: http://stackoverflow.com/questions/38669206/spark-2-0-relative-path-in-absolute-uri-spark-warehouse
spark = SparkSession \
    .builder \
    .appName("MNIST Classifier") \
#    .config('spark.sql.warehouse.dir', 'file:///random/path/as/we/need/to/config/this/but/dont/use/it') \
    .config('spark.executor.instances', 10) \
    .getOrCreate()
    
fileNameTest = './mnist_test.csv'


testData = spark.read.csv(fileNameTest, header=True, inferSchema=True)

deserializedPipeline = PipelineModel.deserializeFromBundle("jar:file:/tmp/pipeline-mnist-classifier-json.zip")

result = deserializedPipeline.transform(testData)
print("Result: " + str(result))
#testprediction = bestModel.transform(testData)
#evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="labelIndex", metricName="f1")
#print("Precision: " + str(evaluator.evaluate(testprediction)))
