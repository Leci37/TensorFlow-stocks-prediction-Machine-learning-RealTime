import pandas as pd
import numpy as np
import re
from sklearn.metrics import classification_report, precision_score

def get_info_model_evaluation(y_pred,y_test,symbol ,indicator_timeframe , retur_info_predict : bool = False ):
    print(y_pred.shape, y_test.shape)
    info_predict = classification_report(y_true=y_test, y_pred=y_pred, digits=3)
    print(info_predict)
    df_info_predict = pd.DataFrame({"y_pred": y_pred, "y_test": y_test})
    # print(f'outputs/model_info/{symbol}_{indicator_timeframe}.csv')
    df_info_predict.to_csv(f'outputs/model_info/{symbol}_{indicator_timeframe}.csv', sep='\t')
    df_details = pd.DataFrame()
    df_details['count'] = df_info_predict.value_counts().to_frame()
    df_details['per%'] = df_info_predict.value_counts(normalize=True).mul(100).round(2)
    # print(df_details.to_string())
    # with open(f'outputs/model_info/{symbol}_{indicator_timeframe}_.info', "w") as text_file:
    #     text_file.write(info_predict + "\n\n\n" + df_details.to_string())
    # print(f'outputs/model_info/{symbol}_{indicator_timeframe}_.info.txt')

    per_sta_pos_f1 = re.search(r" {4,}1 {4,}(0.\d*) {4,}(0.\d*) {4,}(0.\d*)", info_predict)
    per_sta_neg_f1 = re.search(r" {4,}2 {4,}(0.\d*) {4,}(0.\d*) {4,}(0.\d*)", info_predict)
    per_sta_neg_f1_avg = re.search(r" {3,}macro avg {4,}(0.\d*) {4,}(0.\d*) {4,}(0.\d*) {4,}(\d*)", info_predict)
    per_sta_accura = re.search(r" {3,}accuracy {14,}(0.\d*)", info_predict)

    if retur_info_predict:
        return float(per_sta_accura.group(1)), per_sta_pos_f1, per_sta_neg_f1, per_sta_neg_f1_avg
    else:
        return float( per_sta_accura.group(1) ), per_sta_pos_f1, per_sta_neg_f1