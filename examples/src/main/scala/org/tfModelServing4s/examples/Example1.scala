package org.tfModelServing4s
package examples

import scala.util.Try
import tf._
import tf.implicits._
import dsl._
import utils._
import org.tfModelServing4s.utils.show._
import org.tfModelServing4s.utils.show.implicits._

object Example1 {

  def main(args: Array[String]): Unit = {

    val serving = new TFModelServing

    val progr = for {
      _ <- use(serving.load(FileModelSource("/tmp/saved_model_1"), tag = "serve")) { model =>
        for {
          meta        <- serving.metadata(model)
          _           =  println(s"model metadata: $meta")
          signature   <- Try { meta.signatures("serving_default") }
          _           =  println(s"serving signature: $signature")
          _           =  println(s"serving signature inputs: ${signature.inputs}")
          _           =  println(s"serving signature outputs: ${signature.outputs}")

          inputArray  <- Try { Array.range(0, 6).map(_.toFloat) }
          _           =  println(s"input array = ${shows(inputArray)}")

          _ <- use(serving.tensor(inputArray, shape = List(2, 3))) { inputTensor =>
            for {
              inputDef    <- Try { signature.inputs("x") }
              outputDef   <- Try { signature.outputs("z") }
              outputArray <- serving.eval[Array[Array[Float]]](model, outputDef, Map(inputDef -> inputTensor))
              _           =  println(s"output: ${shows(outputArray)}")
            } yield ()
          }

        } yield ()
      }
    } yield ()

    println(s"Program result = $progr")

  }

}
