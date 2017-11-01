package org.tfModelServing4s
package tf

import java.nio.{ByteBuffer, FloatBuffer}

import dsl._
import org.tensorflow.{DataType, Tensor}

object implicits {

  implicit val stringEncoder = new TensorEncoder[Tensor, Array[Byte]] {

    def toTensor(data: Array[Byte], shape: List[Long]) =
      Tensor.create(data)
  }

  implicit val float1DimArrayEncoder = new TensorEncoder[Tensor, Array[Float]] {

    def toTensor(data: Array[Float], shape: List[Long]) =
      Tensor.create(shape.toArray, FloatBuffer.wrap(data))
  }

  implicit val float2DimArrayEncoder = new TensorEncoder[Tensor, Array[Array[Float]]] {

    def toTensor(data: Array[Array[Float]], shape: List[Long]) =
      Tensor.create(shape.toArray, FloatBuffer.wrap(data.flatten))
  }

  implicit val float1DimArrayDecoder = new TensorDecoder[Tensor, Array[Float]] {

    def fromTensor(tensor: Tensor) = {
      val shape = tensor.shape().toList.map(_.toInt)
      val array = Array.ofDim[Float](shape.head)
      tensor.copyTo(array)

      array
    }
  }

  implicit val float2DimArrayDecoder = new TensorDecoder[Tensor, Array[Array[Float]]] {

    def fromTensor(tensor: Tensor) = {
      val shape = tensor.shape().toList.map(_.toInt)
      val array = Array.ofDim[Float](shape.head, shape(1))
      tensor.copyTo(array)

      array
    }
  }

  implicit val closeableTensor = new Closeable[Tensor] {

    def close(resource: Tensor): Unit =  {

      println("releasing TF tensor")
      resource.close()
    }

  }

  implicit val closeableModel = new Closeable[TFModel] {

    def close(resource: TFModel): Unit =  {

      println("closing TF model")
      resource.bundle.close()
    }

  }

}
