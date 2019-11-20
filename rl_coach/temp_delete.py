import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Lambda
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

tfd = tfp.distributions

w0 = 0.125
b0 = 5.
x_range = [-20, 60]

def load_dataset(n=150, n_tst=150):
  np.random.seed(43)
  def s(x):
    g = (x - x_range[0]) / (x_range[1] - x_range[0])
    return 3 * (0.25 + g**2.)
  x = (x_range[1] - x_range[0]) * np.random.rand(n) + x_range[0]
  eps = np.random.randn(n) * s(x)
  y = (w0 * x * (1. + np.sin(x)) + b0) + eps
  x = x[..., np.newaxis]
  x_tst = np.linspace(*x_range, num=n_tst).astype(np.float32)
  x_tst = x_tst[..., np.newaxis]
  return y, x, x_tst

y, x, x_tst = load_dataset()
y = y.reshape(x.shape)
class StdDev(tf.keras.layers.Layer):
    def __init__(self, output_dim=1, **kwargs):
        self.output_dim = output_dim
        super().__init__(**kwargs)
        self.exponential_layer = tf.keras.layers.Lambda(lambda x: tf.exp(x))

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(1,), initializer='zeros', dtype=tf.float32, name='log_std_var')
        super().build(input_shape)

    def call(self, x):
        temp = tf.reduce_mean(x, axis=-1, keepdims=True)
        log_std = temp * 0 + self.bias
        std = self.exponential_layer(log_std)
        return std

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim


#negloglik = lambda y, rv_y: -rv_y.log_prob(y)

def negloglik(y , rv_y):
    loss = -rv_y.log_prob(y)
    #return loss
    return tf.reduce_mean(loss)

def v_loss(y ,b,  value_head):
    old_prob = tfd.MultivariateNormalDiag(loc=[b], scale_diag=[1])
    t = old_prob.log_prob(y*0)
    loss_fn = tf.keras.losses.MeanSquaredError()
    loss = loss_fn(5, value_head)
    return loss




inputs = Input(shape=([1]))
policy_means = Dense(units=1, name="policy_means")(inputs)
policy_stds = tfp.layers.VariableLayer(shape=1, dtype=tf.float32)(inputs)
#policy_stds = StdDev()(inputs)
actions_proba = tfp.layers.DistributionLambda(
    lambda t: tfd.MultivariateNormalDiag(
        loc=t[0], scale_diag=t[1]))([policy_means, policy_stds])

inputs = Input(shape=([1]))
policy_means = Dense(units=1, name="policy_means")(inputs)
policy_stds = tfp.layers.VariableLayer(shape=1, dtype=tf.float32)(inputs)
#policy_stds = StdDev()(inputs)
actions_proba = tfp.layers.DistributionLambda(
    lambda t: tfd.MultivariateNormalDiag(
        loc=t[0], scale_diag=t[1]))([policy_means, policy_stds])



value = Dense(units=1, name="value")(inputs)
model = tf.keras.Model(name='continuous_ppo_head', inputs=inputs, outputs=[value, actions_proba])





#model = tf.keras.Model(name='continuous_ppo_head', inputs=inputs, outputs=actions_proba)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

for epoch in range(1000):
    print(epoch)
    with tf.GradientTape(persistent=True) as tape:
        loss_v = v_loss(y, 8, model(x)[0])
        loss_ppo = negloglik(y, model(x)[1])
        loss = loss_v + loss_ppo
    print(loss_v.numpy())
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
# model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss=negloglik)
# #model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss=[v_loss, negloglik])
# with tf.GradientTape(persistent=True) as tape:
#     loss = negloglik(y, model(x))
#
# gradients = tape.gradient(loss, model.trainable_variables)



#model.fit(x, y, epochs=1000, verbose=False)

# Profit.

[print(np.squeeze(w.numpy())) for w in model.weights]
yhat_ppo = model(x_tst)[1]
print('value head output')
print(model(x_tst)[0])


# model = tf.keras.Sequential([
#   tf.keras.layers.Dense(1),
#   tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)),
# ])
#
# # Do inference.
# model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss=negloglik)
# model.fit(x, y, epochs=10000, verbose=False)
#
# # Profit.
# print('original')
# [print(np.squeeze(w.numpy())) for w in model.weights]


yhat = model(x_tst)
plt.figure(figsize=[6, 1.5])  # inches
plt.plot(x, y, 'b.', label='observed')
#plt.plot(x_tst, yhat.mean(),'r', label='mean', linewidth=4)
plt.plot(x_tst, yhat_ppo.mean(),'--g', label='ppo_mean', linewidth=4)
plt.ylim(-0.,17)
plt.yticks(np.linspace(0, 15, 4)[1:])
plt.xticks(np.linspace(*x_range, num=9))

plt.show()
plt.savefig('foo.png')
a = 1