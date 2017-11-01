import tensorflow as tf
import numpy as np

export_dir = '/tmp/saved_model_1'

builder = tf.saved_model.builder.SavedModelBuilder(export_dir=export_dir)

with tf.Graph().as_default(), tf.Session().as_default() as sess:
    x = tf.placeholder(shape=(2, 3), dtype=tf.float32, name='x')
    y = tf.Variable(np.identity(3), dtype=tf.float32)

    z = tf.matmul(x, y, name='z')

    tf.global_variables_initializer().run()

    zval = z.eval(feed_dict={x: np.random.randn(2, 3)})

    print(zval)

    x_proto_info = tf.saved_model.utils.build_tensor_info(x)
    z_proto_info = tf.saved_model.utils.build_tensor_info(z)

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'x': x_proto_info},
            outputs={'z': z_proto_info},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                         signature_def_map={
                                             tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature
                                         })


builder.save()