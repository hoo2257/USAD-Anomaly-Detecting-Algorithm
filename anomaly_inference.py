import tensorflow as tf
import numpy as np
from USAD_model import *
from utils import gpu_setting, slice_data
import os
import glob
import pandas as pd

gpu_setting(2)

window = 100

model_path = "Models"
encoder_model = "encoder100.h5"
decoder1_model = "decoder1100.h5"
decoder2_model = "decoder2100.h5"

Model = USAD(w_size=6*window, z_size=10)

test = np.zeros((1,600))
test1 = np.zeros((1,10))

Model.encoder(test)
Model.decoder1(test1)
Model.decoder2(test1)

Model.encoder.load_weights(os.path.join(model_path,encoder_model))
Model.decoder1.load_weights(os.path.join(model_path,decoder1_model))
Model.decoder2.load_weights(os.path.join(model_path,decoder2_model))

anomaly_list = glob.glob("scaled_anomaly_resample/*")
normal_list = glob.glob("scaled_normal_resample/*")

df_anomaly = pd.read_csv(anomaly_list[29], index_col=0)
df_normal = pd.read_csv(normal_list[-3], index_col=0)

window_anomaly = slice_data(df_anomaly, window).reshape(-1, 6*window)
window_normal = slice_data(df_normal, window).reshape(-1, 6*window)

alpha = 0.6
beta = 0.4

anomaly_count = 0
false_alarm = open("MissDetection.txt", "w")
anomaly_result = {"mean_anomaly_score":[],"max_anomaly_score":[], "min_anomaly_score":[], "w1_score":[], "w2_score":[]}

for i,file in enumerate(anomaly_list):
    anomaly_score = []
    w1_score = []
    w2_score = []

    df_anomaly = pd.read_csv(anomaly_list[i], index_col=0)
    window_anomaly = slice_data(df_anomaly, window).reshape(-1, 6 * window)
    for a_data in window_anomaly:

        a_data = a_data.reshape(1,6*window)
        w1 = Model.decoder1(Model.encoder(a_data)).numpy()
        w2 = Model.decoder2(Model.encoder(w1)).numpy()

        # anomaly = alpha * (a_data - w1) + beta * (a_data - w2)

        w1_term = a_data - w1
        w2_term = a_data - w2

        anomaly = alpha * (w1_term) + beta * (w2_term)

        t_ = []
        w1_ = []
        w2_ = []

        for t in anomaly:
            # print(np.abs(np.mean(t)))
            t_.append(np.abs(np.mean(t)))
            w1_.append(np.abs(np.mean(w1_term)))
            w2_.append(np.abs(np.mean(w2_term)))
        # print("Anomaly", np.mean(t_))
        anomaly_score.append(np.mean(t_))
        w1_score.append(np.mean(w1_))
        w2_score.append(np.mean(w2_))

    print("MAX Value: {:.3f}/ MIN Value: {:.3f} / MEAN Value: {:.3f}".format(max(anomaly_score), min(anomaly_score), np.mean(anomaly_score)))

    score = np.mean(anomaly_score)
    max_score = np.max(anomaly_score)
    min_score = np.min(anomaly_score)
    avg_w1 = np.mean(w1_score)
    avg_w2 = np.mean(w2_score)

    if score <= 0.04:
        anomaly_count += 1
    else:
        false_alarm.write(file+"\n")

    anomaly_result["max_anomaly_score"].append(max_score)
    anomaly_result["min_anomaly_score"].append(min_score)
    anomaly_result["mean_anomaly_score"].append(score)
    anomaly_result["w1_score"].append(avg_w1)
    anomaly_result["w2_score"].append(avg_w2)

print("{}/{}".format(anomaly_count, len(anomaly_list)))
result = pd.DataFrame.from_dict(anomaly_result)
result.to_csv("inference_anomaly.csv")