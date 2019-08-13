# from tensorflow import keras
#
# import numpy as np
# import tensorflow as tf
#
#
# # # first model without input_dim prints an empty list
# # model = keras.models.Sequential()
# # model.add(keras.layers.Dense(5, weights=[np.ones((3,5)),np.zeros(5)], activation='relu'))
# # print(model.get_weights())
# #
#
# # # second model with input_dim prints the assigned weights
# # model1 = keras.models.Sequential()
# # model1.add(keras.layers.Dense(5,  weights=[np.ones((3,5)),np.zeros(5)], input_dim=3, activation='relu'))
# # model1.add(keras.layers.Dense(1, activation='sigmoid'))
# #
# # print(model1.get_weights())
#
#
#
#
# class ResidualBlock(keras.layers.Layer):
#     def __init__(self, n_layers, n_neurons, **kwargs):
#         super().__init__(**kwargs)
#         self.hidden = [keras.layers.Dense(n_neurons, activation="elu",
#                                           kernel_initializer="he_normal")
#                        for _ in range(n_layers)]
#
#     def call(self, inputs):
#         Z = inputs
#         for layer in self.hidden:
#             Z = layer(Z)
#         return inputs + Z
#
#
# class ResidualRegressor(keras.Model):
#     def __init__(self, outttput_dim, **kwargs):
#         super().__init__(**kwargs)
#         self.hidden1 = keras.layers.Dense(30, input_dim=3, activation="elu",
#                                           kernel_initializer="he_normal")
# #         self.block1 = ResidualBlock(2, 30)
# # #         self.block2 = ResidualBlock(2, 30)
# # #         #self.out = keras.layers.Dense(output_dim)
# # #
# # #     def call(self, inputs):
# # #         Z = self.hidden1(inputs)
# # #         for _ in range(1 + 3):
# # #             Z = self.block1(Z)
# # #         Z = self.block2(Z)
# # #         return Z
# # #         #return self.out(Z)
# # #
# # #
# # # obs = np.array([1., 3., -44., 4.])
# # # obs_batch = tf.expand_dims(obs, 0)
# # # model = ResidualRegressor(3)
# # # model.build(input_shape=(None, 4))
# # # model.get_weights()
# # # model.summary()
# # # a = 1
# # # model(obs_batch)
# #
# #
# # # Adding module path to sys path if not there, so rl_coach submodules can be imported
# # import os
# # import sys
# #
# # module_path = os.path.abspath(os.path.join('..'))
# # resources_path = os.path.abspath(os.path.join('Resources'))
# # if module_path not in sys.path:
# #     sys.path.append(module_path)
# # if resources_path not in sys.path:
# #     sys.path.append(resources_path)
# #
# # from rl_coach.coach import CoachInterface
# #
# # coach = CoachInterface(preset='CartPole_DQN')
# #

a =1
