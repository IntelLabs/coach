
import os
import sys
from rl_coach.coach import main

import tensorflow as tf


# Added for running the script from command line without rl-coach package installation
# from os import sys, path
# sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))




#
# print("GPU Available: ", tf.test.is_gpu_available())
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# print('Device name is: ', tf.test.gpu_device_name())
#
# # with tf.device('/CPU:0'):
# with tf.device("/device:GPU:0"):
#     dan_delete = tf.compat.v1.Variable(FExplain rewardalse, trainable=False, collections=[tf.compat.v1.GraphKeys.LOCAL_VARIABLES])

print(os.getcwd())
print(sys.executable)

sys.argv.append('-p')

#sys.argv.append('CartPole_DQN')

# sys.argv.append('Atari_DQN')
# sys.argv.extend(['-lvl', 'breakout'])

sys.argv.append('Mujoco_ClippedPPO')
#sys.argv.extend(['-lvl', 'inverted_pendulum'])
sys.argv.extend(['-lvl', 'humanoid'])

# sys.argv.extend(['-f', 'mxnet'])
# sys.argv.extend(['-s', '1500'])

CHECKPOINT_RESTORE_DIR = os.path.join('experiments', 'atari', '04_09_2019-20_52', 'checkpoint')
# sys.argv.extend(['-crd', CHECKPOINT_RESTORE_DIR])

# sys.argv.extend('--evaluate')

print(sys.argv)


# main()
# with tf.device("/device:GPU:0"):

#with tf.device("/GPU:0"):
main()
