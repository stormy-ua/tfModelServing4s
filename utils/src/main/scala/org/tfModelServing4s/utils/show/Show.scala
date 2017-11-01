package org.tfModelServing4s.utils
package show

/**
  * Type class defining a string representation of a class.
  * @tparam A Type of a class to define string representation for.
  */
trait Show[A] {

  /**
    * Builds a string representation of an object.
    *
    * @param a An object to build string representation for.
    * @return A string representation of an object.
    */
  def shows(a: A): String

}
