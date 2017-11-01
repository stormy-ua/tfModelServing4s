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

    val modelT = for {
      model            <- serving.load(FileModelSource("/tmp/saved_model_1"), tag = "serve")
      meta             <- serving.metadata(model)
      _                = println(s"model metadata: $meta")
      servingSignature <- Try {
        meta.signatures("serving_default")
      }
      _                = println(s"serving signature: $servingSignature")
      _                = println(s"serving signature inputs: ${servingSignature.inputs}")
      _                = println(s"serving signature outputs: ${servingSignature.outputs}")
    } yield model      -> servingSignature

    if(modelT.isFailure) {
      println(s"model loading failed: $modelT")
    } else {

      val outputT = for {
        (model, signature) <- modelT
        inputArray         <- Try { Array.range(0, 6).map(_.toFloat) }
        _                  =  println(s"input array = ${shows(inputArray)}")
        outputArray        <- use(serving.tensor(inputArray, shape = List(2, 3))) { inputTensor => for {
          inputDef    <- Try { signature.inputs("x") }
          outputDef   <- Try { signature.outputs("z") }
          outputArray <- serving.eval[Array[Array[Float]]](model, outputDef, Map(inputDef -> inputTensor))
          _           =  println(s"output: ${shows(outputArray)}")
        } yield outputArray
        }
      } yield outputArray

      println(outputT)

    }

    for {
      (model, _) <- modelT
      _          <- Try { serving.close(model) }
      _          = println(s"model loaded from '${model.source}' w/ tag '${model.tag}' closed")
    } yield()

  }

}
