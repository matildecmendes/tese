# adicionei

import os
import glob
import numpy as np
import tensorflow.compat.v1 as tf


class LungCTDataset:

    def __init__(self, data_dir, batch_size):

        self.data_dir = data_dir
        self.batch_size = batch_size

        self.num_classes = 1

        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))

        print(f"[LungCTDataset] Found {len(self.files)} files")

    def _load_npy(self, path):

        path = path.decode("utf-8")

        img = np.load(path).astype(np.float32)

        # garantir shape (256,256,1)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)

        return img

    def _tf_load_npy(self, path):

        img = tf.numpy_function(
            self._load_npy,
            [path],
            tf.float32
        )

        img.set_shape([256, 256, 1])

        label = tf.constant(0, dtype=tf.int32)

        return img, label

    def train_input_fn(self, params):

        batch_size = params["batch_size"]

        ds = tf.data.Dataset.from_tensor_slices(self.files)

        ds = ds.shuffle(len(self.files))

        ds = ds.repeat()

        ds = ds.map(
            self._tf_load_npy,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        ds = ds.batch(batch_size, drop_remainder=True)

        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        return ds