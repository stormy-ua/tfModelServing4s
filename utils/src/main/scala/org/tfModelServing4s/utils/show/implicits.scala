package org.tfModelServing4s.utils
package show

object implicits {

  private def arrayToString[A](a: Array[A]): String = a.mkString("[", ", ", "]")

  implicit def toStringShow[A] = new Show[A] {

    def shows(a: A) = a.toString

  }

  implicit def show1DimArray[A](implicit S: Show[A]) = new Show[Array[A]] {

    def shows(a: Array[A]) = arrayToString(a.map(S.shows))

  }

  implicit def show2DimArray[A](implicit S1Dim: Show[Array[A]], S: Show[Array[String]]) = new Show[Array[Array[A]]] {

    def shows(a: Array[Array[A]]) = S.shows(a.map(S1Dim.shows))

  }

}
