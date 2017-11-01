### Reasonable Scala API for serving TensorFlow models

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

Example sources are [here](examples/src/main/scala/org/tfModelServing4s/examples/Example1.scala).