import sbt._

object dependencies {

  def _test     (module: ModuleID): ModuleID = module % "test"
  def _provided (module: ModuleID): ModuleID = module % "provided"

  object Versions {
    val tensorFlow = "1.1.0"
  }

  object tensorFlow {

    val group = "org.tensorflow"

    val core      = group % "tensorflow" % Versions.tensorFlow
    val proto     = group % "proto"      % Versions.tensorFlow

  }

}
