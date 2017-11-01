package org.tfModelServing4s
package examples

import scala.util.Try
import tf._
import tf.implicits._
import dsl._
import utils._
import org.tfModelServing4s.utils.show._
import org.tfModelServing4s.utils.show.implicits._
import java.nio.file.{Files, Paths}

object Example2 {

  private def probsToClass(probs: Array[Float]): String = {
    val classes = io.Source.fromInputStream(getClass.getResourceAsStream("/breeds.csv")).getLines().drop(1).toArray
    val top5 = probs.zip(classes).sortBy { case (prob, idx) => prob }.reverse.take(5)

    top5.mkString("\n")
  }

  def main(args: Array[String]): Unit = {

    val imagePath = args(0)
    val serving = new TFModelServing

    val progr = for {
      _ <- use(serving.load(FileModelSource("/tmp/dogs_1"), tag = "serve")) { model =>
        for {
          meta        <- serving.metadata(model)
          _           =  println(s"model metadata: $meta")
          signature   <- Try { meta.signatures("serving_default") }
          _           =  println(s"serving signature: $signature")
          _           =  println(s"serving signature inputs: ${signature.inputs}")
          _           =  println(s"serving signature outputs: ${signature.outputs}")

          inputArray  <- Try { Array.range(0, 6).map(_.toFloat) }
          _           =  println(s"input array = ${shows(inputArray)}")

          img         <- Try {
            Files.readAllBytes(Paths.get(imagePath))
          }
          _           <- use(serving.tensor(img, shape = List(1))) { inputTensor =>
            for {
              inputDef    <- Try { signature.inputs("image_raw") }
              outputDef   <- Try { signature.outputs("probs") }
              outputArray <- serving.eval[Array[Array[Float]]](model, outputDef, Map(inputDef -> inputTensor))
              _           =  println(s"output: ${shows(outputArray)}")
              clazz       <- Try { probsToClass(outputArray.flatten) }
              _           = println(clazz)
            } yield ()
          }

        } yield ()
      }
    } yield ()

    println(s"Program result = $progr")

  }

}
