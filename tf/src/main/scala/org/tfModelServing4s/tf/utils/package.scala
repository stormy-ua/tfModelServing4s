package org.tfModelServing4s
package tf

import scala.util.Try
import dsl.Closeable

package object utils {

  def use[A, B](resource: Try[A])(f: A => Try[B])(implicit C: Closeable[A]): Try[B] = {
    val result = resource.flatMap(f)

    resource.map(C.close)

    result
  }

}
