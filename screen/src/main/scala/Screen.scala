import java.util.Random

import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.{Matrices, Matrix, Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import utils.VectorUtils

import scala.util.Try

object Screen {
  def main(args: Array[String]) {

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


/*    val sc2 = new SQLContext(sc)
    val df = sc2.createDataFrame(dataRaw.labelNormalizer(p => Row(p.label)), new StructType(Array(StructField("label", DoubleType))))
    val df2 = sc2.createDataFrame(trainingInput.labelNormalizer(p => Row(p.label)), new StructType(Array(StructField("label", DoubleType))))


    df.groupBy("label").count().show()
    df2.groupBy("label").count().show()

    return true
*/

    val rng = new Random(124264398342L)
    // now hash the data
    val hashedDimensions = 20
    //val inputDimensions = 254
    val inputDimensions = 54


    val randn: Matrix = Matrices.randn(hashedDimensions, inputDimensions, rng)
    val bias : Vector = Vectors.zeros(hashedDimensions);//Vectors.dense((0 until hashedDimensions).map({_ => rng.nextGaussian()}).toArray).asInstanceOf[DenseVector]

    val hashed = trainingInput.map(x => {
      val multiply: Vector = randn.multiply(x.features)
      val add = VectorUtils.add(multiply, bias)
      (x, VectorUtils.add(multiply, bias).toDense.values.map(x => if (x > 0) "0" else "1").mkString(""))
    }
    )

    hashed.cache()

    def maxratio(left: Int, right: Int) : Double = {

      Try(Math.max(left,right) / (left + right).toDouble).getOrElse(1d)

/*      val one = if(right > 0) left / right.toDouble else 0
      val two = if(left > 0) right / left.toDouble else 0
      max(one, two)*/
    }

    val procesed = hashed.groupBy(_._2).map({case (hash, it) =>
      // aggregate
      var pos : Int = 0
      var neg : Int = 0
      it.foreach( point => if (point._1.label == 1) pos+=1 else neg +=1 )

      (hash, pos, neg)
    })


    val filtered = procesed.filter(p => {val m = maxratio(p._2, p._3); (m < 0.95) && (m > 0.6)})
      //.filter(p => p._2 > 1 && p._3 > 1)
      .map(_._1).collect().toSet

    // 887
   // hashed.take(10) foreach println

    val hist = procesed.map(p => maxratio(p._2, p._3)).histogram(20)

    val training = hashed.filter(p => filtered.contains(p._2)).map(_._1)
    hashed.unpersist()

   // use the
    // val training = trainingInput//.sample(withReplacement = false, fraction = 0.0381380952380952380952380952381, seed = 133)

    // Run training algorithm to build the model
    val numIterations = 100
    val model = SVMWithSGD.train(training, numIterations)
    //val model = SVMWithSGD.train(trainingInput, numIterations)

    // Clear the default threshold.
    model.clearThreshold()

    // Compute raw scores on the test set.
    val scoreAndLabels = test.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }

    val supports = training.filter(point => {
      val score = model.predict(point.features)

      val predictedLabel = if (score > 0) 1.0 else -1.0
      val x = predictedLabel * (if (point.label == 1.0) 1 else -1)

      (x * score <= 1) && (x * score >= 0)
    }
    ).count()



    // Get evaluation metrics.
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()

    val counts = training.count()

    println(s"Area under ROC = $auROC, count = $counts with supports = $supports\n")

    println(hist._1.mkString("\n"))
    println(hist._2.mkString("\n"))

  }

}