from USAD_model import *
from utils import gpu_setting, slice_data
import os
import numpy as np
import pandas as pd

class USAD_inference():

    def __init__(self, window, threshold=0.04, model_path="Models", enc_path="encoder100.h5",
                 dec1_path="decoder1100.h5", dec2_path = "decoder2100.h5"):

        gpu_setting(2)
        self.window = window
        self.model = USAD(w_size=6*self.window, z_size=10)
        self.thres = threshold

        dummy_enc = np.zeros((1,600))
        dummy_dec = np.zeros((1,10))

        self.model.encoder(dummy_enc)
        self.model.decoder1(dummy_dec)
        self.model.decoder2(dummy_dec)

        self.model.encoder.load_weights(os.path.join(model_path, enc_path))
        self.model.decoder1.load_weights(os.path.join(model_path, dec1_path))
        self.model.decoder2.load_weights(os.path.join(model_path, dec2_path))

        self.alpha = 0.9
        self.beta = 0.1
        self.thres = threshold

    def get_test_data(self, file):
        """
        1 shift event data to sliced window
        :param file: .csv file directory
        :return: sliced data
        """
        df = pd.read_csv(file, index_col=0)
        test_data = slice_data(df, self.window).reshape(-1, 6*self.window)
        return test_data

    def run_USAD(self, file, value):
        """
        Perform Unsupervised Anomaly Detection for single .csv file(Shift Event)
        :param file: .csv file directory
        :param value: value used for calculating final anomaly score
        :return: Detection results and data for logging
        """

        anomaly_scores = []
        w1_scores = []
        w2_scores = []
        window_anomaly = self.get_test_data(file)

        for a_input in window_anomaly:
            a_input = a_input.reshape(1, 6*self.window)
            w1 = self.model.decoder1(self.model.encoder(a_input)).numpy()
            w2 = self.model.decoder2(self.model.encoder(w1)).numpy()

            w1_term = np.sqrt(np.mean(a_input - w1) ** 2)
            w2_term = np.sqrt(np.mean(a_input - w2) ** 2)

            anomaly = self.alpha*(w1_term) + self.beta*(w2_term) # anomaly score for each sliced window data

            anomaly_scores.append(anomaly)
            w1_scores.append(w1_term)
            w2_scores.append(w2_term)

        # Final Anomaly score, this score will be compared with predefined threshold value for anomaly detection
        score = value[0] * max(anomaly_scores) + value[1] * np.mean(anomaly_scores)
        max_score = np.max(anomaly_scores)
        min_score = np.min(anomaly_scores)

        return score, max_score, min_score, w1_scores, w2_scores, anomaly_scores





