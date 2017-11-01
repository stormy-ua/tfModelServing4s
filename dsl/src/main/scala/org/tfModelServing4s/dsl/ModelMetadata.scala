package org.tfModelServing4s.dsl

/**
  * A model description.
  *
  * @param signatures Maps a model signature name to its metadata.
  */
case class ModelMetadata(signatures: Map[String, SignatureMetadata])
