package hash

import org.apache.spark.mllib.linalg.{DenseMatrix, DenseVector, Matrix}

import scala.util.Random

object Generator {

  def generateRandomGaussianMatrix(random: Random, rows: Int, columns: Int, w: Double): (Matrix, DenseVector) = {

    val size = 8 // Double is 64 bit

    val values = new Array[Double](rows * columns)
    def i = 0

    for( i <- 0 to rows * columns){
      values(i) = random.nextGaussian
    }

    val bias = new Array[Double](rows)

    for( i <- 0 to rows){
        bias(i) = random.nextDouble * w
    }

    (new DenseMatrix(rows, columns, values), new DenseVector(bias))
  }
}
