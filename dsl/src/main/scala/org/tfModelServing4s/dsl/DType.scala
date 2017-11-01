package org.tfModelServing4s.dsl

/**
  Tensor data type
 */
sealed trait DType

case object DTypeFloat  extends DType
case object DTypeDouble extends DType
case object DTypeInt32  extends DType
case object DTypeUInt8  extends DType
case object DTypeString extends DType
case object DTypeInt64  extends DType
case object DTypeBool   extends DType
