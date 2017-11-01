package org.tfModelServing4s.utils

package object show {

  def shows[A](a: A)(implicit S: Show[A]): String = S.shows(a)

}
