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

# refer to this post for more details: http://stackoverflow.com/questions/38669206/spark-2-0-relative-path-in-absolute-uri-spark-warehouse
spark = SparkSession \
    .builder \
    .appName("MNIST Classifier") \
#    .config('spark.sql.warehouse.dir', 'file:///random/path/as/we/need/to/config/this/but/dont/use/it') \
    .config('spark.executor.instances', 10) \
    .getOrCreate()

fileNameTrain = './mnist_train.csv'
fileNameTest = './mnist_test.csv'
# read the data from raw csv file.
# #We use "inferSchema" as "True" since we want to import the data as integers. Otherwise spark will treat it as strings.
mnist_train = spark.read.csv(fileNameTrain, header=True, inferSchema=True)

testData = spark.read.csv(fileNameTest, header=True, inferSchema=True)
# total 784 features
FEATURE_NUM = 784
# assemble those features to a vector to consume in Spark
assembler = VectorAssembler(
    inputCols=["x{0}".format(i) for i in range(FEATURE_NUM)],
    outputCol="features")

# Transform pixel0,pixel1...pixel783 to one column named "features"
labeledPoints = assembler.transform(mnist_train).select("label", "features")

# turn label into an index for later classification use
# StringIndexer encodes a string column of labels to a column of label indices. The indices are in [0, numLabels), ordered by label frequencies, so the most frequent label gets index 0.
# More details here: https://spark.apache.org/docs/latest/ml-features.html#stringindexer
indexer = StringIndexer(inputCol="label", outputCol="labelIndex")
labeledPoints = indexer.fit(labeledPoints).transform(labeledPoints)
[trainData, testData] = labeledPoints.randomSplit([0.8, 0.2])
# labeledPoints.printSchema()

# define the classifier here
rfc = RandomForestClassifier(labelCol="labelIndex", featuresCol="features", impurity='gini', maxBins=32)

pipeline = Pipeline(stages=[rfc])

# define the param grids to search for best hyper-parameters
paramGrid = ParamGridBuilder() \
    .addGrid(rfc.numTrees, range(3, 3)) \
    .addGrid(rfc.maxDepth, range(4, 4)) \
    .build()

# Define the cross validator. We need to define a model to be validated, an evaluator which is used to evaluate the model, and a param grid to be searched for
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(),
                          numFolds=1)

# Run TrainValidationSplit, and choose the best set of parameters.
start_time = time.time()
# we can now fit the model using above configurations
cvModel = crossval.fit(trainData)
print("grid search time --- %s seconds ---" % (time.time() - start_time))

# then we can choose the best model, either save it for later usage, or use it to train the test dataset.
bestModel = cvModel.bestModel
# bestModel.save('bestModel')
# We can print out the best model to see more details.

sparkTransformed = bestModel.transform(trainData)
bestModel.serializeToBundle("jar:file:/tmp/pipeline_bundle.zip", sparkTransformed)
#testprediction = bestModel.transform(testData)
#evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="labelIndex", metricName="f1")
#print("Precision: " + str(evaluator.evaluate(testprediction)))
