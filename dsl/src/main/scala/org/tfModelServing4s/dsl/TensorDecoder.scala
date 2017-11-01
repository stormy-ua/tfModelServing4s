package org.tfModelServing4s.dsl

/**
  * Converts a tensor to its representation in the form of a data structure.
  * @tparam TTensor Type of tensor to convert from.
  * @tparam TRepr Type of the representation to convert to e.g. Array, List etc.
  */
trait TensorDecoder[TTensor, TRepr] {

  def fromTensor(tensor: TTensor): TRepr

}
