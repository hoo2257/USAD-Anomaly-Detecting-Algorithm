import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers

class Encoder(Model):
    def __init__(self, in_size, latent_size):
        super().__init__()
        self.enc1 = layers.Dense(512, activation="relu", input_shape=(in_size,), name="en1")
        self.enc2 = layers.Dense(512, activation="relu", name="en2")
        self.enc3 = layers.Dense(256, activation="relu", name="en3")
        self.enc4 = layers.Dense(256, activation="relu", name="en4")
        self.enc5 = layers.Dense(128, activation="relu", name="en5")
        self.enc6 = layers.Dense(128, activation="relu", name="en6")
        self.enc7 = layers.Dense(64, activation="relu", name="en7")
        self.enc8 = layers.Dense(64, activation="relu", name="en8")
        self.enc9 = layers.Dense(32, activation="relu", name="en9")
        self.enc10 = layers.Dense(32, activation="relu", name="en10")

        self.enc11 = layers.Dense(latent_size, activation="relu", name="en11")

        self._set_inputs(tf.TensorSpec([None, in_size], tf.float32, name='inputs'))

    def call(self, x):
        out = self.enc1(x)
        out = self.enc2(out)
        out = self.enc3(out)
        out = self.enc4(out)
        out = self.enc5(out)
        out = self.enc6(out)
        out = self.enc7(out)
        out = self.enc8(out)
        out = self.enc9(out)
        out = self.enc10(out)
        z = self.enc11(out)

        return z

class Decoder(Model):
    def __init__(self, latent_size, out_size):
        super().__init__()
        self.dec1 = layers.Dense(32, activation="relu", input_shape=(latent_size,), name="de1_1")
        self.dec2 = layers.Dense(32, activation="relu", name="de1_2")
        self.dec3 = layers.Dense(64, activation="relu", name="de1_3")
        self.dec4 = layers.Dense(64, activation="relu", name="de1_4")
        self.dec5 = layers.Dense(128, activation="relu", name="de1_5")
        self.dec6 = layers.Dense(128, activation="relu", name="de1_6")
        self.dec7 = layers.Dense(256, activation="relu", name="de1_7")
        self.dec8 = layers.Dense(256, activation="relu", name="de1_8")
        self.dec9 = layers.Dense(512, activation="relu", name="de1_9")
        self.dec10 = layers.Dense(512, activation="relu", name="de1_10")

        self.dec11 = layers.Dense(out_size, activation="sigmoid", name="de1_11")

    def call(self, z):
        out = self.dec1(z)
        out = self.dec2(out)
        out = self.dec3(out)
        out = self.dec4(out)
        out = self.dec5(out)
        out = self.dec6(out)
        out = self.dec7(out)
        out = self.dec8(out)
        out = self.dec9(out)
        out = self.dec10(out)

        w = self.dec11(out)

        return w

class USAD(Model):
    def __init__(self, w_size, z_size):
        super().__init__()
        self.encoder = Encoder(w_size, z_size)
        self.decoder1 = Decoder(z_size, w_size)
        self.decoder2 = Decoder(z_size, w_size)

    def call(self, batch):
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))

        return w1, w2, w3







