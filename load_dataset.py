import tensorflow as tf
import numpy as np

class data_generator(tf.keras.utils.Sequence):

    def __init__(self,batch_size, data):
        self.data = data
        self.batch_size = batch_size
        self.l_data = data.shape[0]
        self.on_epoch_end()

    def __len__(self):
        return self.l_data // self.batch_size

    def __getitem__(self, idx):
        batch_idx = self.idxs[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch =  self.data[batch_idx]

        return batch

    def on_epoch_end(self):
        self.idxs = np.random.permutation(self.l_data)



