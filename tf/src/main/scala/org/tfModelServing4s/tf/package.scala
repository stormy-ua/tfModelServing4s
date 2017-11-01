package org.tfModelServing4s

import dsl._
import org.tensorflow.framework.{DataType => FDataType}

package object tf {

  /**
    * Maps a TensorFlow data type to the serving model algebra data type.
    */
  private[tf] val dTypesMap = Map(
    FDataType.DT_BOOL    -> DTypeBool,
    FDataType.DT_DOUBLE  -> DTypeDouble,
    FDataType.DT_FLOAT   -> DTypeFloat,
    FDataType.DT_INT32   -> DTypeInt32,
    FDataType.DT_INT64   -> DTypeInt64,
    FDataType.DT_STRING  -> DTypeString,
    FDataType.DT_BOOL    -> DTypeUInt8
  )

}
