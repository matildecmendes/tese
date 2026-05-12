import tensorflow.compat.v1 as tf

print("TF:", tf.__version__)
print("GPU:", tf.test.is_gpu_available())

print(tf.config.experimental.list_physical_devices('GPU'))