scalaVersion in Global := "2.12.4"

lazy val dsl_ = Project(id = "dsl", base = file("dsl"))
lazy val tf = project.dependsOn(dsl_ % "test->test;compile->compile", utils % "test->test;compile->compile")
lazy val examples = project.dependsOn(dsl_ % "test->test;compile->compile",
  utils % "test->test;compile->compile",
  tf % "test->test;compile->compile")

lazy val utils = project

lazy val tfModelServing4s  = project.in(file(".")).aggregate(dsl_, utils, tf, examples)

