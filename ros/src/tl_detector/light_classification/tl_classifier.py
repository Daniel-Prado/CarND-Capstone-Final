# from styx_msgs.msg import TrafficLight
import os
import tensorflow as tf
import inception_resnet_v2
import rospy


class TLClassifier(object):

    def __init__(self):
        # TODO load classifier
        self._image_size = [800, 600]
        self.sess = tf.InteractiveSession()
        self.input_tensor, self.output_tensors = self._model()
        checkpoint_dir = os.path.join(os.path.dirname(__file__), "models")
        self._restore_model(checkpoint_dir)

    def _model(self):
        image_input = tf.placeholder(tf.uint8, shape=self._image_size + [3])
        image = self._preprocess_for_eval(image_input)
        images = tf.expand_dims(image, 0)
        logits, _ = inception_resnet_v2.inception_resnet_v2(images, num_classes=4, is_training=False)
        prediction = tf.squeeze(tf.argmax(logits, 1))
        score = tf.squeeze(tf.reduce_max(logits, 1))
        return image_input, (prediction, score)

    def _restore_model(self, checkpoint_dir):
        save_path = tf.train.latest_checkpoint(checkpoint_dir)
        saver = tf.train.Saver()
        saver.restore(self.sess, save_path)

    def _preprocess_for_eval(self, image, height=None, width=None, scope=None):
        """Prepare one image for evaluation.

        If height and width are specified it would output an image with that size by
        applying resize_bilinear.

        If central_fraction is specified it would crop the central fraction of the
        input image.

        Args:
          image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
            [0, 1], otherwise it would converted to tf.float32 assuming that the range
            is [0, MAX], where MAX is largest positive representable number for
            int(8/16/32) data type (see `tf.image.convert_image_dtype` for details)
          height: integer
          width: integer
          central_fraction: Optional Float, fraction of the image to crop.
          scope: Optional scope for name_scope.
        Returns:
          3-D float Tensor of prepared image.
        """
        with tf.name_scope(scope, 'eval_image', [image, height, width]):
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)

            if height and width:
                image = tf.image.resize_images(image,
                                               [height, width],
                                               # method=tf.image.ResizeMethod.AREA,
                                               align_corners=False)
            image = tf.subtract(image, 0.5)
            image = tf.multiply(image, 2.0)
            return image

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # TODO implement light color prediction
        prediction, score = self.sess.run(self.output_tensors, feed_dict={self.input_tensor: image})
        prediction = prediction if prediction < 4 else 4
        rospy.logwarn("TLClassifier prediction\t%s\tscore\t%s", prediction, score)
        return prediction
