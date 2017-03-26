package utils

import breeze.linalg.{Vector => BV}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf

/** expose vector.toBreeze and Vectors.fromBreeze
  */
object VectorUtils {

  def fromBreeze(breezeVector: BV[Double]): Vector = {
    Vectors.dense(breezeVector.toArray)
  }

  def asBreeze(vector: Vector): BV[Double] = {
    // this is vector.asBreeze in Spark 2.0
    BV(vector.toArray)
  }

  val addVectors: UserDefinedFunction = udf {
    (v1: Vector, v2: Vector) => fromBreeze( asBreeze(v1) + asBreeze(v2) )
  }

  def add(v1: Vector, v2: Vector): Vector = fromBreeze( asBreeze(v1) + asBreeze(v2))

}