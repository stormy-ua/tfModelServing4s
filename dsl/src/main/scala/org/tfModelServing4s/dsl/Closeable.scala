package org.tfModelServing4s.dsl

/**
  * Type class representing a resource that has to be released/closed explicitly
  * e.g. an un-managed resource like a file handle or a database connection.
  *
  * @tparam A Type of the resource that has to be closed explicitly.
  */
trait Closeable[A] {

  def close(resource: A): Unit

}
