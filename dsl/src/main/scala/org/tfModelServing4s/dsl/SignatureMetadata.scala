package org.tfModelServing4s.dsl

/**
  * Describes a model signature. A model signature is the set of
  * input tensors together w/ output tensors. An output tensor
  * could be calculated using the model and by feeding input tensors.
  * In other words, a signature could be considered as a definition
  * of a function which uses a model to produce its output based on inputs.
  *
  * @param name
  * @param inputs
  * @param outputs
  */
case class SignatureMetadata(name: String, inputs: Map[String, TensorMetadata], outputs: Map[String, TensorMetadata])
