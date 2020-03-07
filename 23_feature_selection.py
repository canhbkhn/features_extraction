import os, pickle, re, collections
import pandas as pd
import numpy as np
#
import matplotlib.pyplot as plt
from IPython.display import display

from lib_modeling import *
from xgboost import plot_importance

# some settings for displaying Pandas results
pd.set_option('display.width', 2000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.precision', 4)
pd.set_option('display.max_colwidth', -1)

version = "v07"


pdf_features_label = pd.read_csv("pdf_features_label.csv.bz2", compression="bz2")
#pdf_features_label = pd.read_csv('pdf_features_label.csv')#os.path.join("/home/stack/Data_science /data/", "pdf_features_label.csv.bz2"), compression="bz2")
#meta_cols = ["SK_ID_CURR", "TARGET", "tvt_code"]
meta_cols = ["id", "label", "tvt_code"]
ls_features = [cname for cname in pdf_features_label.columns if cname not in meta_cols]

#
print("Number of features: {}".format(len(ls_features)))
print(pdf_features_label.shape)
display(pdf_features_label.head().T)
# get train/val/test index 
# list
idx_train = pdf_features_label["tvt_code"] == "train"
idx_test_list = [pdf_features_label["tvt_code"] == "val", pdf_features_label["tvt_code"] == "test"]

#
param_init = {
    "objective": "binary:logistic",
    "booster": "gbtree",
    "max_depth": 4,  # default: 3 only for depthwise
    "n_estimators": 1500,  # default: 500
    "learning_rate": 0.025,  # default: 0.05
    "subsample": 0.7,
    "colsample_bytree": 0.6,  # default:  1.0
    "colsample_bylevel": 0.5,  # default: 1.0

    #
    "silent": True,
    "n_jobs": 16,

    #
    "tree_method": "hist",  # default: auto
    "grow_policy": "lossguide",  # default depthwise
}

param_fit = {
    "eval_metric": "auc",
    "early_stopping_rounds": 500,  # default: 100
    "verbose": 200,
}

options = {
    # turn to filter features
    "nturn": 4,

    # turn to run random state
    "auc_check_per_turn_n": 3,

    # drop per turn
    "ndrop_per_turn": 10
}

ls_res_selection_info = feature_selection_steps(
    pdf_input=pdf_features_label,
    ls_features=ls_features,
    #target_name="TARGET",
    target_name="label",
    target_posval=0,
    idx_train=idx_train,
    idx_test_list=idx_test_list,
    xgb_param_init=param_init,
    xgb_param_fit=param_fit,
    options=options,
)


i_max_turn = max(range(options["nturn"]), key=lambda i_res: ls_res_selection_info[i_res]["auc"][-1])
xgb_model_i = ls_res_selection_info[i_max_turn]["model"]
ls_auc_i = ls_res_selection_info[i_max_turn]["auc"]
ls_imp_i = ls_res_selection_info[i_max_turn]["imp"]
print("AUC: {}".format(ls_auc_i))


# save model to file
res_model = ls_res_selection_info[i_max_turn]
res_model["features"] = res_model["model"].get_booster().feature_names
pickle.dump(res_model, open("xgb_model_{}.mod".format(version), "wb"))


# read model
with open("xgb_model_{}.mod".format(version), "rb") as input_file:
    res_model = pickle.load(input_file)
print(res_model.keys())
# #
visualize_auc(pdf_features_label, "test", res_model)
#
fig_height = len(res_model["imp"]) / 4
fig, ax = plt.subplots(figsize=(10, fig_height))
plot_importance(res_model["model"], ax=ax)
plt.show()








