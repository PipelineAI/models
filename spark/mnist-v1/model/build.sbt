val globalSettings = Seq(
  version := "1.0",
  scalaVersion := "2.11.8"
)

//-Dhttp.proxyHost=your.proxy.server
//-Dhttp.proxyPort=8080
//-Dhttps.proxyHost=your.proxy.server
//-Dhttps.proxyPort=8080

sourcesInBase := false
scalaSource in Compile := baseDirectory.value 
javaSource in Compile := baseDirectory.value 

lazy val settings = (project in file("."))
                    .settings(name := "mnist-jvm")
                    .settings(globalSettings:_*)
                    .settings(libraryDependencies ++= deps)

val mleapVersion = "0.12.0"
val sparkVersion = "2.3.2"

lazy val deps = Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "ml.combust.mleap" %% "mleap-core" % mleapVersion,
  "ml.combust.mleap" %% "mleap-runtime" % mleapVersion,
  "ml.combust.mleap" %% "mleap-base" % mleapVersion,
  "ml.combust.mleap" %% "mleap-spark" % mleapVersion,
  "ml.combust.mleap" %% "mleap-spark-base" % mleapVersion
)
