import tensorflow as tf

class VGG19_model:
  def __init__(self):
    selected_layers = ["block3_conv4", "block4_conv4", "block5_conv4"]
    self.vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
    self.vgg.trainable = False
    self.outputs = [self.vgg.get_layer(l).output for l in selected_layers]

  def get_vgg19(self):
    vgg_model = tf.keras.Model(self.vgg.input, self.outputs)
    return vgg_model