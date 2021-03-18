import os
from tensorflow.python import pywrap_tensorflow

checkpoint_path = os.path.join('./experiments/kevin_test/checkpoint', "model.ckpt-67")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)  # tf.train.NewCheckpointReader
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
    # print(reader.get_tensor(key))
