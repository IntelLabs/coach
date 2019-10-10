# # import tensorflow as tf
# # import tensorflow_probability as tfp
# # tfd = tfp.distributions
# # import numpy as np
# #
# # tfd = tfp.distributions
# #
# #
# #
#
# # from __future__ import absolute_import
# # from __future__ import division
# # from __future__ import print_function
#
# from pprint import pprint
# import matplotlib.pyplot as plt
# import numpy as np
# #import seaborn as sns
#
# # import tensorflow as tf
# # from tensorflow.python import tf2
# # if not tf2.enabled():
# #   import tensorflow.compat.v2 as tf
# #   tf.enable_v2_behavior()
# #   assert tf2.enabled()
#
#
# import tensorflow as tf
# #tf.config.gpu.set_per_process_memory_fraction(0.4)
# import tensorflow_probability as tfp
#
#
#
# negloglik = lambda y, rv_y: -rv_y.log_prob(y)
#
# #sns.reset_defaults()
# #sns.set_style('whitegrid')
# #sns.set_context('talk')
# #sns.set_context(context='talk',font_scale=0.7)
#
# #@title Synthesize dataset.
# w0 = 0.125
# b0 = 5.
# x_range = [-20, 60]
#
# def load_dataset(n=150, n_tst=150):
#   np.random.seed(43)
#   def s(x):
#     g = (x - x_range[0]) / (x_range[1] - x_range[0])
#     return 3 * (0.25 + g**2.)
#   x = (x_range[1] - x_range[0]) * np.random.rand(n) + x_range[0]
#   eps = np.random.randn(n) * s(x)
#   y = (w0 * x * (1. + np.sin(x)) + b0) + eps
#   x = x[..., np.newaxis]
#   x_tst = np.linspace(*x_range, num=n_tst).astype(np.float32)
#   x_tst = x_tst[..., np.newaxis]
#   return y, x, x_tst
#
# y, x, x_tst = load_dataset()
#
# tfd = tfp.distributions
#
#
#
#
# # Build model.
# model = tf.keras.Sequential([
#   tf.keras.layers.Dense(1),
#   tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)),
# ])
#
# # Do inference.
# model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss=negloglik)
# model.fit(x, y, epochs=1000, verbose=False)
#
# # Profit.
# [print(np.squeeze(w.numpy())) for w in model.weights]
# yhat = model(x_tst)
# assert isinstance(yhat, tfd.Distribution)





import matplotlib.pyplot as plt
import tensorflow as tf
layers = tf.keras.layers
import numpy as np
print(tf.__version__)

mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.figure(figsize=(10,10))
for i in range(25):
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(x_train[i], cmap=plt.cm.binary)
  plt.xlabel(class_names[y_train[i]])
plt.show()

model = tf.keras.Sequential()
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])


tf.keras.utils.plot_model(model)
a = tf.keras.utils.model_to_dot(model).create(prog='dot', format='svg')
model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test)
