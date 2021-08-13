from USAD_model import *
from load_dataset import *
from utils import gpu_setting
import numpy as np
import tensorflow as tf
import os
import pandas as pd
import json

gpu_setting(5)
batch_size = 64
window = 100
train_data = np.load("train_data/test/train_data_{}.npy".format(window))
print(train_data.shape)
train_data = train_data.reshape(-1,6*window)
val_data = np.load("train_data/test/validation_data_{}.npy".format(window))
val_data = val_data.reshape(-1,6*window)
losses = {"val1_loss": [], "val2_loss": [], "train1_loss": [], "train2_loss": []}

train_loss1 = []
train_loss2 = []

train_dataset = data_generator(batch_size, train_data)
val_dataset = data_generator(batch_size, val_data)

Model = USAD(w_size=6*window, z_size=10)
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
epochs = 20
#
for epoch in range(1, epochs+1):
    print("Epoch {}/{}".format(epoch, epochs))
    for step, x_batch_train in enumerate(train_dataset):

        with tf.GradientTape() as tape:
            x_batch_train = tf.cast(x_batch_train, tf.float32)
            _,w2,w3  = Model(x_batch_train)
            loss1 = (1 / epoch) * tf.reduce_mean((x_batch_train - w2) ** 2) + (1 - 1 / epoch) * tf.reduce_mean((x_batch_train - w3) ** 2)

        grad_ae1 = tape.gradient(loss1, Model.trainable_variables)
        optimizer.apply_gradients(zip(grad_ae1, Model.trainable_variables))
        mean_loss1 = tf.reduce_mean(tf.identity(loss1)).numpy()
        train_loss1.append(mean_loss1)


        with tf.GradientTape() as tape:
            w1,_,w3  = Model(x_batch_train)
            loss2 = (1 / epoch) * tf.reduce_mean((x_batch_train - w1) ** 2) - (1 - 1 / epoch) * tf.reduce_mean((x_batch_train - w3) ** 2)

        grad_ae2 = tape.gradient(loss2, Model.trainable_variables)
        optimizer.apply_gradients(zip(grad_ae2, Model.trainable_variables))
        mean_loss2 = tf.reduce_mean(tf.identity(loss2)).numpy()
        train_loss2.append(mean_loss2)


        print("\r", f"Epochs:{epoch}/{epochs}", "file: ",
              f"{step}/{len(train_dataset)}",
              "*" * (30 * step // len(train_dataset)), "-" * (30 * (len(train_dataset) - step) // len(train_dataset)),
              "Loss1: ", loss1, "Loss2: ", loss2, end="")


    print("")
    losses["train1_loss"].append(np.mean(train_loss1))
    losses["train2_loss"].append(np.mean(train_loss2))

    # print(np.mean(train_loss1))
    # print(np.mean(train_loss2))

    val_losses1 = []
    val_losses2 = []
    for idx, x_batch_val in enumerate(val_dataset):
        w1,w2,w3  = Model(x_batch_val)
        val_loss1 = 1 / epoch * tf.reduce_mean((x_batch_val - w1) ** 2) + (1 - 1 / epoch) * tf.reduce_mean(
            (x_batch_val - w3) ** 2)
        val_loss2 = 1 / epoch * tf.reduce_mean((x_batch_val - w2) ** 2) - (1 - 1 / epoch) * tf.reduce_mean(
            (x_batch_val - w3) ** 2)

        val_losses1.append(val_loss1.numpy())
        val_losses2.append(val_loss2.numpy())

        print("\rValidation","."*(idx%10), end="")

    print("")
    val1_loss = np.mean(val_losses1)
    val2_loss = np.mean(val_losses2)

    print("val1_loss: ", val1_loss, "val2_loss: ", val2_loss)
    losses["val1_loss"].append(val2_loss)
    losses["val2_loss"].append(val2_loss)


model_path = "Models"
if not os.path.isdir(model_path):
    os.mkdir(model_path)

Model.encoder.save_weights("{}/encoder{}.h5".format(model_path, window))
Model.decoder1.save_weights("{}/decoder1{}.h5".format(model_path, window))
Model.decoder2.save_weights("{}/decoder2{}.h5".format(model_path, window))

# print(losses)
a = pd.DataFrame.from_dict(losses)
a.to_csv("result.csv")







