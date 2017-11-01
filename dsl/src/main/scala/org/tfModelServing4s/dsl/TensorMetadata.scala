package org.tfModelServing4s.dsl

/**
  * Describes a tensor properties.
  * @param name Name of the tensor.
  * @param opName Name of the operation the tensor is produced by.
  * @param dataType Data type of the tensor.
  * @param shape The shape of the tensor.
  */
case class TensorMetadata(name: String, opName: String, dataType: DType, shape: List[Long])
