
import com.databricks.spark.avro._
// Import the rest of the required packages
import org.apache.spark.ml.mleap.feature.OneHotEncoder
import org.apache.spark.ml.feature.{StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.{RandomForestRegressor, LinearRegression}
import org.apache.spark.ml.{Pipeline, PipelineStage}

// MLeap/Bundle.ML Serialization Libraries
import ml.combust.mleap.spark.SparkSupport._
import resource._
import ml.combust.bundle.BundleFile
import org.apache.spark.ml.bundle.SparkBundleContext

// MLeap/Bundle.ML De-Serialization Libraries
import ml.combust.mleap.runtime.MleapContext.defaultContext
import ml.combust.mleap.runtime.MleapSupport._
import java.io.File


import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoder}

val inputFile = "file:////tmp/airbnb.avro"

var dataset = spark.sqlContext.read.format("com.databricks.spark.avro").
  load(inputFile)

var datasetFiltered = dataset.filter("price >= 50 AND price <= 750 and bathrooms > 0.0")
println(dataset.count())
println(datasetFiltered.count())

datasetFiltered.registerTempTable("df")

val datasetImputed = spark.sqlContext.sql(f"""
    select
        id,
        city,
        case when state in('NY', 'CA', 'London', 'Berlin', 'TX' ,'IL', 'OR', 'DC', 'WA')
            then state
            else 'Other'
        end as state,
        space,
        price,
        bathrooms,
        bedrooms,
        room_type,
        host_is_superhost,
        cancellation_policy,
        case when security_deposit is null
            then 0.0
            else security_deposit
        end as security_deposit,
        price_per_bedroom,
        case when number_of_reviews is null
            then 0.0
            else number_of_reviews
        end as number_of_reviews,
        case when extra_people is null
            then 0.0
            else extra_people
        end as extra_people,
        instant_bookable,
        case when cleaning_fee is null
            then 0.0
            else cleaning_fee
        end as cleaning_fee,
        case when review_scores_rating is null
            then 80.0
            else review_scores_rating
        end as review_scores_rating,
        case when square_feet is not null and square_feet > 100
            then square_feet
            when (square_feet is null or square_feet <=100) and (bedrooms is null or bedrooms = 0)
            then 350.0
            else 380 * bedrooms
        end as square_feet
    from df
    where bedrooms is not null
""")


datasetImputed.select("square_feet", "price", "bedrooms", "bathrooms", "cleaning_fee").describe().show()

// Most popular cities (original dataset)

spark.sqlContext.sql(f"""
    select 
        state,
        count(*) as n,
        cast(avg(price) as decimal(12,2)) as avg_price,
        max(price) as max_price
    from df
    group by state
    order by count(*) desc
""").show()

// Most expensive popular cities (original dataset)
dataset.registerTempTable("df")

spark.sqlContext.sql(f"""
    select 
        city,
        count(*) as n,
        cast(avg(price) as decimal(12,2)) as avg_price,
        max(price) as max_price
    from df
    group by city
    order by avg(price) desc
""").filter("n>25").show()

// Step 2. Create our feature pipeline and train it on the entire dataset
val continuousFeatures = Array("bathrooms",
  "bedrooms",
  "security_deposit",
  "cleaning_fee",
  "extra_people",
  "number_of_reviews",
  "square_feet",
  "review_scores_rating")

val categoricalFeatures = Array("room_type",
  "host_is_superhost",
  "cancellation_policy",
  "instant_bookable",
  "state")

val allFeatures = continuousFeatures.union(categoricalFeatures)

// Filter all null values
val allCols = allFeatures.union(Seq("price")).map(datasetImputed.col)
val nullFilter = allCols.map(_.isNotNull).reduce(_ && _)
val datasetImputedFiltered = datasetImputed.select(allCols: _*).filter(nullFilter).persist()

println(datasetImputedFiltered.count())

val Array(trainingDataset, validationDataset) = datasetImputedFiltered.randomSplit(Array(0.7, 0.3))

val continuousFeatureAssembler = new VectorAssembler(uid = "continuous_feature_assembler").
    setInputCols(continuousFeatures).
    setOutputCol("unscaled_continuous_features")

val continuousFeatureScaler = new StandardScaler(uid = "continuous_feature_scaler").
    setInputCol("unscaled_continuous_features").
    setOutputCol("scaled_continuous_features")

val categoricalFeatureIndexers = categoricalFeatures.map {
    feature => new StringIndexer(uid = s"string_indexer_$feature").
      setInputCol(feature).
      setOutputCol(s"${feature}_index")
}
val categoricalFeatureOneHotEncoders = categoricalFeatureIndexers.map {
    indexer => new OneHotEncoder(uid = s"oh_encoder_${indexer.getOutputCol}").
      setInputCol(indexer.getOutputCol).
      setOutputCol(s"${indexer.getOutputCol}_oh")
}

val featureColsRf = categoricalFeatureIndexers.map(_.getOutputCol).union(Seq("scaled_continuous_features"))
val featureColsLr = categoricalFeatureOneHotEncoders.map(_.getOutputCol).union(Seq("scaled_continuous_features"))

// assemble all processes categorical and continuous features into a single feature vector
val featureAssemblerLr = new VectorAssembler(uid = "feature_assembler_lr").
    setInputCols(featureColsLr).
    setOutputCol("features_lr")
val featureAssemblerRf = new VectorAssembler(uid = "feature_assembler_rf").
    setInputCols(featureColsRf).
    setOutputCol("features_rf")

val estimators: Array[PipelineStage] = Array(continuousFeatureAssembler, continuousFeatureScaler).
    union(categoricalFeatureIndexers).
    union(categoricalFeatureOneHotEncoders).
    union(Seq(featureAssemblerLr, featureAssemblerRf))

val featurePipeline = new Pipeline(uid = "feature_pipeline").
    setStages(estimators)

val sparkFeaturePipelineModel = featurePipeline.fit(datasetImputedFiltered)

println("Finished constructing the pipeline")

// Create our random forest model
val randomForest = new RandomForestRegressor(uid = "random_forest_regression").
    setFeaturesCol("features_rf").
    setLabelCol("price").
    setPredictionCol("price_prediction")

val sparkPipelineEstimatorRf = new Pipeline().setStages(Array(sparkFeaturePipelineModel, randomForest))
val sparkPipelineRf = sparkPipelineEstimatorRf.fit(datasetImputedFiltered)

println("Complete: Training Random Forest")

// Create our linear regression model
val linearRegression = new LinearRegression(uid = "linear_regression").
    setFeaturesCol("features_lr").
    setLabelCol("price").
    setPredictionCol("price_prediction")

val sparkPipelineEstimatorLr = new Pipeline().setStages(Array(sparkFeaturePipelineModel, linearRegression))
val sparkPipelineLr = sparkPipelineEstimatorLr.fit(datasetImputedFiltered)

println("Complete: Training Linear Regression")

val sbc = SparkBundleContext().withDataset(sparkPipelineLr.transform(datasetImputedFiltered))
for(bf <- managed(BundleFile("jar:file:/tmp/airbnb.model.lr.zip"))) {
        sparkPipelineLr.writeBundle.save(bf)(sbc).get
      }

val sbcRf = SparkBundleContext().withDataset(sparkPipelineRf.transform(datasetImputedFiltered))
for(bf <- managed(BundleFile("jar:file:/tmp/airbnb.model.rf.zip"))) {
        sparkPipelineRf.writeBundle.save(bf)(sbcRf).get
      }

val mleapTransformerLr = (for(bf <- managed(BundleFile("jar:file:/tmp/airbnb.model.lr.zip"))) yield {
      bf.loadMleapBundle().get.root
    }).tried.get

val mleapTransformerRf = (for(bf <- managed(BundleFile("jar:file:/tmp/airbnb.model.rf.zip"))) yield {
      bf.loadMleapBundle().get.root
    }).tried.get

import ml.combust.mleap.runtime.serialization.FrameReader

val s = scala.io.Source.fromURL("https://s3-us-west-2.amazonaws.com/mleap-demo/frame.json").mkString

println(s)

val bytes = s.getBytes("UTF-8")

for(frame <- FrameReader("ml.combust.mleap.json").fromBytes(bytes);
    frameLr <- mleapTransformerLr.transform(frame);
    frameLrSelect <- frameLr.select("price_prediction");
    frameRf <- mleapTransformerRf.transform(frame);
    frameRfSelect <- frameRf.select("price_prediction")) {
      println("Price LR: " + frameLrSelect.dataset(0).getDouble(0))
      println("Price RF: " + frameRfSelect.dataset(0).getDouble(0))
}

import ml.combust.mleap.spark.SparkSupport._

val inputFile = "file:////tmp/airbnb.avro"

var dataset = spark.sqlContext.read.format("com.databricks.spark.avro").
  load(inputFile)

var datasetFiltered = dataset.filter("price >= 50 AND price <= 750 and bathrooms > 0.0")

val continuousFeatures = Array("bathrooms",
  "bedrooms",
  "security_deposit",
  "cleaning_fee",
  "extra_people",
  "number_of_reviews",
  "square_feet",
  "review_scores_rating")

val categoricalFeatures = Array("room_type",
  "host_is_superhost",
  "cancellation_policy",
  "instant_bookable",
  "state")

val allFeatures = continuousFeatures.union(categoricalFeatures)

datasetFiltered.registerTempTable("df")

val datasetImputed = spark.sqlContext.sql(f"""
    select
        id,
        city,
        case when state in('NY', 'CA', 'London', 'Berlin', 'TX' ,'IL', 'OR', 'DC', 'WA')
            then state
            else 'Other'
        end as state,
        space,
        price,
        bathrooms,
        bedrooms,
        room_type,
        host_is_superhost,
        cancellation_policy,
        case when security_deposit is null
            then 0.0
            else security_deposit
        end as security_deposit,
        price_per_bedroom,
        case when number_of_reviews is null
            then 0.0
            else number_of_reviews
        end as number_of_reviews,
        case when extra_people is null
            then 0.0
            else extra_people
        end as extra_people,
        instant_bookable,
        case when cleaning_fee is null
            then 0.0
            else cleaning_fee
        end as cleaning_fee,
        case when review_scores_rating is null
            then 80.0
            else review_scores_rating
        end as review_scores_rating,
        case when square_feet is not null and square_feet > 100
            then square_feet
            when (square_feet is null or square_feet <=100) and (bedrooms is null or bedrooms = 0)
            then 350.0
            else 380 * bedrooms
        end as square_feet
    from df
    where bedrooms is not null
""")


datasetImputed.select("square_feet", "price", "bedrooms", "bathrooms", "cleaning_fee").describe().show()

val allCols = allFeatures.union(Seq("price")).map(datasetImputed.col)
val nullFilter = allCols.map(_.isNotNull).reduce(_ && _)
val datasetImputedFiltered = datasetImputed.select(allCols: _*).filter(nullFilter).persist()

// Use your Spark ML Pipeline to transform the Spark DataFrame
val transformedDataset = sparkTransformer.transform(datasetImputedFiltered)

// Create a custom SparkBundleContext and provide the transformed DataFrame
implicit val sbc = SparkBundleContext().withDataset(transformedDataset)

// Serialize the pipeline as you would normally
(for(bf <- managed(BundleFile(file))) yield {
  sparkTransformer.writeBundle.save(bf).get
}).tried.get

val sparkDataframe = mleapTransformerLr.sparkTransform(transformedDataset)

sparkDataframe.columns

sparkDataframe.select("bedrooms", "bathrooms", "price", "price_prediction").show(10)

// Libraries to deploy models to combust cloud (optional)
import akka.actor.ActorSystem
import akka.stream.ActorMaterializer
import scala.concurrent.duration._
import scala.concurrent.Await
implicit val system = ActorSystem("combust-client")
implicit val materializer = ActorMaterializer()

{
    implicit val context = sbc
    Await.result(sparkPipelineLr.deploy("http://localhost:65327", "my_username", "my_model_lr"), 10.seconds)
}
{
    implicit val context = sbcRf
    Await.result(sparkPipelineRf.deploy("http://localhost:65327", "my_username", "my_model_rf"), 10.seconds)
}
/*
import ml.combust.model.core.domain.v1._
import scala.concurrent.duration._
import scala.concurrent.Await
import ml.combust.mleap.runtime.serialization.FrameReader


val client = ml.combust.model.client.Client("http://localhost:65327")

val s = scala.io.Source.fromURL("https://s3-us-west-2.amazonaws.com/mleap-demo/frame.json").mkString
val bytes = s.getBytes("UTF-8")

var leapFrame = FrameReader("ml.combust.mleap.json").fromBytes(bytes).get
val result = Await.result(client.transform(TransformRequest().withUsername("my_username").withModelId("my_model_lr").withFrame(leapFrame)), 10.seconds)
val leapFrameFromServer = result.frame
leapFrameFromServer

leapFrameFromServer.select("price_prediction").get.dataset(0).getDouble(0)
*/
