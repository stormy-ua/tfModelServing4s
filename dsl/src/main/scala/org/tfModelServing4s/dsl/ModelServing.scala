package org.tfModelServing4s.dsl

/**
  * Abstract model serving algebra.
  * Defines a set of operations on a pre-built model that is stored
  * somewhere on an external storage.
  *
  * @tparam F Type of the effect used by this algebra.
  */
trait ModelServing[F[_]] {

  // Model type.
  type TModel

  // Tensor type.
  type TTensor

  /**
    * Loads a model from an external source.
    *
    * @param source A source to load a model from.
    * @param tag A tag that could be used to uniquely identify
    *            a concrete computational graph inside a saved model.
    * @return A model.
    */
  def load(source: ModelSource, tag: String): F[TModel]

  /**
    * Queries a model metadata.
    *
    * @param model A model to query metadata for.
    * @return A model metadata wrapped into an effect.
    */
  def metadata(model: TModel): F[ModelMetadata]

  /**
    * Creates a tensor from its representation in the form of a different data structure
    * e.g. Array, List etc.
    *
    * @param data An instance of a data structure for building the tensor.
    * @param shape Shape of the resulting tensor.
    * @param E Tensor encoder to use.
    * @tparam TRepr Type of the data structure to build tensor from.
    * @return A tensor wrapped into an effect.
    */
  def tensor[TRepr](data: TRepr, shape: List[Long])(implicit E: TensorEncoder[TTensor, TRepr]): F[TTensor]

  /**
    * Evaluates a tensor value by feeding a set of input tensors as defined by a signature
    * into the model.
    * @param model A model to use.
    * @param output Output tensor metadata.
    * @param feed Map of a tensor metadata to the tensor to use as an input.
    * @param D Tensor decoder to use.
    * @param C Closeable type class for tensor type being used.
    * @tparam TRepr Type of the data structure to map output tensor to.
    * @return A data structure representing an output tensor calculated by feeding a set
    *         of inputs into the model.
    */
  def eval[TRepr](model: TModel, output: TensorMetadata, feed: Map[TensorMetadata, TTensor])
                 (implicit D: TensorDecoder[TTensor, TRepr], C: Closeable[TTensor]): F[TRepr]

  /**
    * Closes a resource and releases all related resources.
    *
    * @param resource A resource to close.
    * @return Unit.
    */
  def close[A](resource: A)(implicit C: Closeable[A]): F[Unit]
}
