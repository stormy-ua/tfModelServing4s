package org.tfModelServing4s.dsl

/**
  * Builds a tensor from its representation in the form of a data structure.
  *
  * @tparam TTensor Type of the tensor to build.
  * @tparam TRepr Type of the representation to build tensor from e.g. Array, List etc.
  */
trait TensorEncoder[TTensor, TRepr] {

  def toTensor(data: TRepr, shape: List[Long]): TTensor

}