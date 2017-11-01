### Reasonable Scala API for serving TensorFlow models

#### API

The core API algebra is agnostic to TensorFlow itself and could be used for binding to any TensorFlow-like library:

```scala
/**
  * Abstract model serving algebra.
  * Defines a set of operations on a pre-built model that is stored
  * somewhere on an external storage.
  *
  * @tparam F Type of the effect used by this algebra.
  */
trait ModelServing[F[_]] {

  // Model type.
  type TModel

  // Tensor type.
  type TTensor

  /**
    * Loads a model from an external source.
    *
    * @param source A source to load a model from.
    * @param tag A tag that could be used to uniquely identify
    *            a concrete computational graph inside a saved model.
    * @return A model.
    */
  def load(source: ModelSource, tag: String): F[TModel]

  /**
    * Queries a model metadata.
    *
    * @param model A model to query metadata for.
    * @return A model metadata wrapped into an effect.
    */
  def metadata(model: TModel): F[ModelMetadata]

  /**
    * Creates a tensor from its representation in the form of a different data structure
    * e.g. Array, List etc.
    *
    * @param data An instance of a data structure for building the tensor.
    * @param shape Shape of the resulting tensor.
    * @param E Tensor encoder to use.
    * @tparam TRepr Type of the data structure to build tensor from.
    * @return A tensor wrapped into an effect.
    */
  def tensor[TRepr](data: TRepr, shape: List[Long])(implicit E: TensorEncoder[TTensor, TRepr]): F[TTensor]

  /**
    * Evaluates a tensor value by feeding a set of input tensors as defined by a signature
    * into the model.
    * @param model A model to use.
    * @param output Output tensor metadata.
    * @param feed Map of a tensor metadata to the tensor to use as an input.
    * @param D Tensor decoder to use.
    * @param C Closeable type class for tensor type being used.
    * @tparam TRepr Type of the data structure to map output tensor to.
    * @return A data structure representing an output tensor calculated by feeding a set
    *         of inputs into the model.
    */
  def eval[TRepr](model: TModel, output: TensorMetadata, feed: Map[TensorMetadata, TTensor])
                 (implicit D: TensorDecoder[TTensor, TRepr], C: Closeable[TTensor]): F[TRepr]

  /**
    * Closes the model and releases all related resources.
    *
    * @param model A model to close.
    * @return Unit.
    */
  def close(model: TModel): F[Unit]
}
```

The binding for TensorFlow is implemented in [TFModelServing.scala](tf/src/main/scala/org/tfModelServing4s/tf/TFModelServing.scala).

#### Example #1

1. Create a TensorFlow graph and save it using [Saved Model API](https://www.tensorflow.org/programmers_guide/saved_model):

```python
import tensorflow as tf
import numpy as np

export_dir = '/tmp/saved_model_1'

builder = tf.saved_model.builder.SavedModelBuilder(export_dir=export_dir)

with tf.Graph().as_default(), tf.Session().as_default() as sess:
    x = tf.placeholder(shape=(2, 3), dtype=tf.float32, name='x')
    y = tf.Variable(np.identity(3), dtype=tf.float32)

    z = tf.matmul(x, y, name='z')

    tf.global_variables_initializer().run()

    zval = z.eval(feed_dict={x: np.random.randn(2, 3)})

    print(zval)

    x_proto_info = tf.saved_model.utils.build_tensor_info(x)
    z_proto_info = tf.saved_model.utils.build_tensor_info(z)

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'x': x_proto_info},
            outputs={'z': z_proto_info},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                         signature_def_map={
                                             tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature
                                         })


builder.save()
```

The script is available [here](examples/src/main/python/example1.py).

In this example the computational graph simply does matrix multiplication of an input [2x3] matrix and identity [3x3] matrix. The result matrix has to be equal to the input matrix.
The model with all variables will be saved into `/tmp/saved_model_1` dir.

2. Load saved model in Scala, read its metadata, feed input matrix into the loaded computational graph and show the output matrix:

```scala
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
```

Output:

```text
model metadata: ModelMetadata(Map(serving_default -> SignatureMetadata(tensorflow/serving/predict,Map(x -> TensorMetadata(x:0,x,DTypeFloat,List(2, 3))),Map(z -> TensorMetadata(z:0,z,DTypeFloat,List(2, 3))))))
serving signature: SignatureMetadata(tensorflow/serving/predict,Map(x -> TensorMetadata(x:0,x,DTypeFloat,List(2, 3))),Map(z -> TensorMetadata(z:0,z,DTypeFloat,List(2, 3))))
serving signature inputs: Map(x -> TensorMetadata(x:0,x,DTypeFloat,List(2, 3)))
serving signature outputs: Map(z -> TensorMetadata(z:0,z,DTypeFloat,List(2, 3)))
input array = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
releasing TF tensor
output: [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
releasing TF tensor
closing TF model
Program result = Success(())
```

The example sources are [here](examples/src/main/scala/org/tfModelServing4s/examples/Example1.scala).

#### Example #2 (Dog Breed Classification)

[This repo](https://github.com/stormy-ua/dog-breeds-classification) contains the description how to build dog breed classifier using pre-trained Inception model. 
The final model gets exported as a "frozen" graph instead of a SavedModel format, though. [This script](https://github.com/stormy-ua/dog-breeds-classification/blob/master/src/freezing/frozen_to_saved_model.py) converts frozen model to SavedModel and could be used with this library.

1. Follow steps to build dog breed classifier as described [here](https://github.com/stormy-ua/dog-breeds-classification/blob/master/README.md)
2. Convert resulting "frozen" model to SavedModel format: 
`python -m src.freezing.frozen_to_saved_model`
3. Use the dog breed classifier saved model to classify an arbitrary dog image:

```scala
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
```

Here is the output for the [sample dog image](examples/src/main/resources/airedale.jpg): 

```text 
(0.9796268,3,airedale)
(0.00985535,83,otterhound)
(0.007160045,57,irish_terrier)
(0.0015072817,28,chesapeake_bay_retriever)
(8.2421233E-4,118,wire-haired_fox_terrier)
```

The model outputs probabilities array which gets mapped into top 5 breeds together with their probabilities. 

The example source code is [here](examples/src/main/scala/org/tfModelServing4s/examples/Example2.scala).