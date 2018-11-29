import ml.combust.bundle.BundleFile
import ml.combust.mleap.runtime.MleapSupport._

import ml.combust.mleap.runtime.serialization.FrameReader

import resource._
// https://github.com/combust/mleap/pull/286
import ml.combust.mleap.runtime.MleapSupport
import ml.combust.bundle.BundleFile

// MLeap/Bundle.ML Serialization Libraries
import ml.combust.mleap.spark.SparkSupport._
import resource._
import ml.combust.bundle.BundleFile
import org.apache.spark.ml.bundle.SparkBundleContext

import ml.combust.mleap.runtime.MleapContext.defaultContext
import java.io.File

import org.apache.spark.ml.mleap.SparkUtil
import ml.combust.mleap.runtime.frame.FrameBuilder
import ml.combust.mleap.runtime.frame.DefaultLeapFrame

object pipeline_invoke {

  def main(args: Array[String]): Unit = {
    // load the Spark pipeline we saved in the previous section
    // There is a wonky limitation of BundleFile that requires
    //   this filename parameter to start at the /root/directory

    val mleapPipeline = (for(bf <- managed(BundleFile("jar:file:/tmp/pipeline_bundle.zip"))) yield {
      bf.loadMleapBundle().get.root
    }).tried.get

    val s = scala.io.Source.fromFile("pipeline_test_request.json").mkString
    val bytes = s.getBytes("UTF-8")

    val frame = FrameReader("ml.combust.mleap.json").fromBytes(bytes)
    val transformed_frame = mleapPipeline.transform(frame.get)
    val data = transformed_frame.get.dataset
    println(data)
  }
}
