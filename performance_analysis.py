from USAD_Inferecne import *
from utils import colored
import glob
import numpy as np


class performace_analysis():

    def __init__(self, window=100):

        self.scale = [(1.0,0),(0.9,0.1),(0.8,0.2),(0.7,0.3),(0.6,0.4),(0.5,0.5),(0.4,0.6),(0.3,0.7),(0.2,0.8),(0.1,0.9),(0,1.0)]
        self.window = window
        self.usad = USAD_inference(window=self.window)

    def detect_anomaly(self,value, mode, thres=0.04):

        mode_list = ["anomaly", "normal"]

        if mode not in mode_list:
            print(colored(255,0,0, "[Error] Invalid Keyword Input"))
            raise ValueError

        if mode == "anomaly":
            anomaly_list = glob.glob("scaled_anomaly_resample/*")

            # TODO Additional work is required for logger
            logger = open("MissDetection.txt_{}_{}".format(value[0], value[1]), "w")
            detection_result = {"mean_anomaly_score": [], "max_anomaly_score": [],
                              "min_anomaly_score": [], "w1_score": [], "w2_score": []}
        elif mode == "normal":
            anomaly_list = glob.glob("scaled_normal_resample/*")

            # TODO Additional work is required for logger
            logger = open("FalseAlarm.txt_{}_{}".format(value[0], value[1]), "w")
            detection_result = {"mean_anomaly_score": [], "max_anomaly_score": [],
                              "min_anomaly_score": [], "w1_score": [], "w2_score": []}

        count = 0
        for i, file in enumerate(anomaly_list):
            score, max_score, min_score, w1_scores, w2_scores, anomaly_scores = self.usad.run_USAD(file, value)

            if mode == "anomaly":
                if score <= thres:
                    count += 1
                else:
                    logger.write(file+"\n"+str(score)+"\n")

            elif mode == "normal":
                if score > thres:
                    count += 1
                else:
                    # print(score)
                    logger.write(file + "\n" + str(score) + "\n")


            # TODO Additional work is required for logger
            detection_result["max_anomaly_score"].append(max_score)
            detection_result["min_anomaly_score"].append(min_score)
            detection_result["mean_anomaly_score"].append(score)


        miss = len(anomaly_list) - count
        # result = pd.DataFrame.from_dict(anomaly_result)
        # result.to_csv("inference_{}_{}_{}.csv".format(mode,value[0], value[1]))

        return count, miss


    def value_analysis(self, modes = ("anomaly", "normal")):
        f1_score_list = []
        precision_list = []
        recall_list = []
        accuracy_list = []
        for value in self.scale:

            TP,FN = self.detect_anomaly(value, modes[0])
            TN,FP = self.detect_anomaly(value, modes[1])

            recall = (TP)/(TP+FN)
            precision = (TP)/(TP + FP)
            f1_score = 2*(precision*recall)/(precision + recall)
            accuracy = (TP + TN)/(TP+FN+FP+TN)

            recall_list.append(recall)
            precision_list.append(precision)
            f1_score_list.append(f1_score)
            accuracy_list.append(accuracy)

        max_f1_score = max(f1_score_list)
        f1_idx = f1_score_list.index(max(f1_score_list))
        best_value = self.scale[f1_idx]
        best_acc = accuracy_list[f1_idx]

        print("Max F1 Score : {} / Best Value : {} / Accuracy : {}"
              .format(max_f1_score, best_value, best_acc))

        return best_value

    def get_roc_curve(self, best_value, modes = ("anomaly", "normal")):
        # TODO Additional work is needed
        FPR_list = []
        TPR_list = []

        thres_list = np.arange(0.01,0.11,0.01)

        for t in thres_list:
            TP, FN = self.detect_anomaly(best_value, modes[0], thres=t)
            TN, FP = self.detect_anomaly(best_value, modes[1], thres=t)

            FPR = TP/(TP+FN)
            TPR = FP/(FP+TN)

            FPR_list.append(FPR)
            TPR_list.append(TPR)

        return FPR_list, TPR_list, thres_list


if __name__ == "__main__":
    evaluate = performace_analysis()
    best_value = evaluate.value_analysis()
    FPRs, TPRs, thresholds = evaluate.get_roc_curve(best_value)
