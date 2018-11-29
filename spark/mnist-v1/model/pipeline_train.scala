// Derived from the following:
//    https://github.com/combust/mleap/wiki/Serializing-a-Spark-ML-Pipeline-and-Scoring-with-MLeap
//    https://github.com/combust/mleap-docs/blob/master/demos/mnist.md

// Note that we are taking advantage of com.databricks:spark-csv package to load the data
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,IndexToString, Binarizer}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator}
import org.apache.spark.ml.{Pipeline,PipelineModel}  
import org.apache.spark.ml.feature.PCA

import ml.combust.mleap.runtime.MleapSupport
import org.apache.spark.sql.SparkSession

// MLeap/Bundle.ML Serialization Libraries
import ml.combust.mleap.spark.SparkSupport._
import resource._
import ml.combust.bundle.BundleFile
import org.apache.spark.ml.bundle.SparkBundleContext

import ml.combust.mleap.runtime.MleapSupport._
import ml.combust.mleap.runtime.MleapContext.defaultContext
import java.io.File

import org.apache.spark.ml.mleap.SparkUtil


// For implicit conversions like converting RDDs to DataFrames
// import spark.implicits._

object pipeline_train {

def main(args: Array[String]): Unit = {
  val spark = SparkSession
   .builder()
// 2 threads doesn't seem to work well
   .master("local[4]")
   .appName("Spark")
   .getOrCreate()

  val datasetPath = "./mnist_train.csv"
  var dataset = spark.sqlContext.read.format("com.databricks.spark.csv").
                 option("header", "true").
                 option("inferSchema", "true").
                 load(datasetPath)
                 
  val testDatasetPath = "./mnist_test.csv"
  var test = spark.sqlContext.read.format("com.databricks.spark.csv").
                 option("inferSchema", "true").
                 option("header", "true").
                 load(testDatasetPath)

  // Define Dependent and Independent Features
  val predictionCol = "label"
  val labels = Seq("0","1","2","3","4","5","6","7","8","9")  
  val pixelFeatures = (0 until 784).map(x => s"x$x").toArray

  val layers = Array[Int](pixelFeatures.length, 784, 800, labels.length)

  val vector_assembler = new VectorAssembler()  
    .setInputCols(pixelFeatures)
    .setOutputCol("features")

  val stringIndexer = { 
    new StringIndexer()
  //val stringIndexer = {
  //    new IndexToString()
      .setInputCol(predictionCol)
      .setOutputCol("label_index")
      .fit(dataset)
  }
  
   val rstringIndexer = {
      new IndexToString()
      .setInputCol(predictionCol)
      .setOutputCol("label_index")
  }
  
  val binarizer = new Binarizer()  
    .setInputCol(vector_assembler.getOutputCol)
    .setThreshold(127.5)
    .setOutputCol("binarized_features")
  
  val pca = new PCA().
    setInputCol(binarizer.getOutputCol).
    setOutputCol("pcaFeatures").
    setK(10)

  val featurePipeline = new Pipeline().setStages(Array(vector_assembler, stringIndexer, binarizer, pca))

  // Transform the raw data with the feature pipeline and persist it
  val featureModel = featurePipeline.fit(dataset)

  val datasetWithFeatures = featureModel.transform(dataset)

  // Select only the data needed for training and persist it
  val datasetPcaFeaturesOnly = datasetWithFeatures.select(rstringIndexer.getOutputCol, pca.getOutputCol)
  val datasetPcaFeaturesOnlyPersisted = datasetPcaFeaturesOnly.persist()

  val rf = new RandomForestClassifier().
      setFeaturesCol(pca.getOutputCol).
      setLabelCol(rstringIndexer.getOutputCol).
      setPredictionCol("prediction").
      setProbabilityCol("probability").
      setRawPredictionCol("raw_prediction")

  val rfModel = rf.fit(datasetPcaFeaturesOnlyPersisted)

  val pipeline = SparkUtil.createPipelineModel(uid = "pipeline", Array(featureModel, rfModel))

  val sbc = SparkBundleContext().withDataset(rfModel.transform(datasetWithFeatures))

  // There is a wonky limitation of BundleFile that requires 
  //   this filename parameter to start at the /root/directory
  // Also, it doesn't let you overwrite it
  for (bf <- managed(BundleFile("jar:file:/tmp/pipeline_bundle.zip"))) {
        pipeline.writeBundle.save(bf)(sbc).get
  }
 }
}

