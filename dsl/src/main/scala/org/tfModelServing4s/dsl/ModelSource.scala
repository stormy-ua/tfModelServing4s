package org.tfModelServing4s.dsl

/**
  Defines a source to get a model from.
 */
sealed trait ModelSource

/**
  * Carries a path to a model on the file system.
  * @param path Path to a model on the file system.
  */
case class FileModelSource(path: String) extends ModelSource
