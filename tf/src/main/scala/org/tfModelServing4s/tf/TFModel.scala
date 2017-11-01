package org.tfModelServing4s
package tf

import org.tensorflow.SavedModelBundle
import org.tensorflow.framework.MetaGraphDef
import dsl._

/**
  * TensorFlow saved model representation.
  *
  * @param source A source the model is loaded from.
  * @param tag A tag which uniquely identifies a computational graph defined in the saved model.
  * @param bundle TensorFlow API object representing the saved model bundle.
  * @param graphDef TensorFlow API object representing a computational graph.
  */
case class TFModel(source: ModelSource, tag: String, bundle: SavedModelBundle, graphDef: MetaGraphDef)