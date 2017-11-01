package org.tfModelServing4s
package tf

import org.tensorflow.{SavedModelBundle, Tensor}
import org.tensorflow.framework.{MetaGraphDef, SignatureDef, TensorInfo}
import scala.collection.JavaConverters._
import scala.util.Try
import dsl._

/**
  * Model serving algebra implementation for TensorFlow saved model.
  */
class TFModelServing extends ModelServing[Try] {

  type TModel = TFModel
  type TTensor = Tensor

  private def pathForSource(source: ModelSource): String = source match {
    case FileModelSource(path) => path
  }

  def load(source: ModelSource, tag: String): Try[TModel] = Try {
    val bundle = SavedModelBundle.load(pathForSource(source), tag)
    val graphDef = MetaGraphDef.parseFrom(bundle.metaGraphDef())

    TFModel(source, tag, bundle, graphDef)
  }

  private def parseTensorInfo(opName: String, tensorInfo: TensorInfo) = {

    val shape = tensorInfo.getTensorShape.getDimList.asScala.map(_.getSize).toList
    val name = tensorInfo.getName
    val opName = name.split(":").head


    TensorMetadata(name = name,
      opName = opName,
      dataType = dTypesMap(tensorInfo.getDtype),
      shape = shape)
  }

  private def parseSignatureDef(signatureDef: SignatureDef) = {
    val inputs = signatureDef.getInputsMap.asScala.map {
      case (k, v) => k -> parseTensorInfo(k, v)
    }.toMap
    val outputs = signatureDef.getOutputsMap.asScala.map {
      case (k, v) => k -> parseTensorInfo(k, v)
    }.toMap

    SignatureMetadata(signatureDef.getMethodName, inputs = inputs, outputs = outputs)
  }

  def metadata(model: TModel): Try[ModelMetadata] = Try {
    val signatures = model.graphDef.getSignatureDefMap.asScala.mapValues(parseSignatureDef).toMap

    ModelMetadata(signatures)
  }

  private def run(model: TModel, output: TensorMetadata, feed: Map[TensorMetadata, TTensor]): Try[List[TTensor]] = Try {
    val runner = feed.foldLeft(model.bundle.session.runner) { case (r, (tensorMeta, tensor)) =>
      r.feed(tensorMeta.opName, tensor)
    }

    runner.fetch(output.opName).run().asScala.toList
  }

  def eval[TRepr](model: TModel, output: TensorMetadata, feed: Map[TensorMetadata, TTensor])
                 (implicit D: TensorDecoder[Tensor, TRepr], C: Closeable[TTensor]): Try[TRepr] = {
    val tensorT = for {
      ts <- run(model, output, feed)
      t <- Try(ts.head)
    } yield t

    val reprT = tensorT.map(D.fromTensor)

    tensorT.map(C.close)

    reprT
  }

  def close(model: TModel): Try[Unit] = Try {
    model.bundle.session().close()
  }

  def tensor[TRepr](data: TRepr, shape: List[Long])(implicit E: TensorEncoder[TTensor, TRepr]): Try[TTensor] = Try {
    E.toTensor(data, shape)
  }

}