package random

import java.util.Locale

import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

class RandomScreening {

  def run(): Unit = {
    val conf = new SparkConf()
      .setAppName("Simple Application")
      .setMaster("local[*]")

    val sc = new SparkContext(conf)

    //Generator.generateRandomGaussianMatrix(new Random(10), 10, 10, 1)

    // remap to fit into 0 and 1 space
    val labelNormalizer = Map(-1.0 -> 0, 1.0 -> 1, 2.0 -> 0)

    val dataRaw: RDD[LabeledPoint] = MLUtils
      //.loadLibSVMFile(sc, "../data/webspam_wc_normalized_unigram/*")
      .loadLibSVMFile(sc, "../data/covtype.libsvm.binary.scale/covtype.libsvm.binary.scale")
      .map {
        case LabeledPoint(category, features) => LabeledPoint(labelNormalizer(category), features)
      }

    // scale the data
    val scaler = new StandardScaler(withMean = true, withStd = true).fit(dataRaw.map(x => x.features))

    val data = dataRaw.map({
      case LabeledPoint(category, features) => LabeledPoint(category, scaler.transform(Vectors.dense(features.toArray)))
    })

    // Split data into training (60%) and test (40%).
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    val trainingInput = splits(0).cache()
    val test = splits(1)


    // now random sample

    //val samples =
    val res = scala.collection.mutable.ArrayBuffer.empty[(Double, Double, Long)]

    for (fraction <- Array(0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)) {

      val sample = trainingInput.sample(withReplacement = false, fraction, seed = 212121142114L)

      // Run training algorithm to build the model
      val numIterations = 100
      val model = SVMWithSGD.train(sample, numIterations)
      //val model = SVMWithSGD.train(trainingInput, numIterations)

      // Clear the default threshold.
      model.clearThreshold()

      // Compute raw scores on the test set.
      val scoreAndLabels = test.map { point =>
        val score = model.predict(point.features)
        (score, point.label)
      }

      // Get evaluation metrics.
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      val auROC = metrics.areaUnderROC()

      val counts = sample.count()

      println(s"Area under ROC = $auROC, count = $counts \n")

      res += ((fraction, auROC, counts))

    }

    // now print
    println("Fraction\tAUROC\tcount")
    val output = res.map(element => "%f\t%f\t%d".formatLocal(Locale.ENGLISH, element._1, element._2,element._3)).mkString("\n")
    println(output)




  }

}
