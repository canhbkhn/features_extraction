# Full width
from IPython.core.display import display, HTML
#display(HTML("<style>.container { width:100% !important; }</style>"))
import math
import os
import subprocess
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

#
from lib_modeling import *
from lib_feature_engineering import *

# some settings for displaying Pandas results
pd.set_option('display.width', 2000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.precision', 4)
pd.set_option('display.max_colwidth', -1)
version = "v07"

#pdf_features_label = pd.read_csv(os.path.join("/home/stack/Data_science /data/", "pdf_features_label.csv.bz2"), compression="bz2")
pdf_features_label = pd.read_csv("pdf_features_label.csv.bz2", compression="bz2")
pdf_train = pdf_features_label.query("tvt_code == 'train'")
#meta_cols = ["SK_ID_CURR", "TARGET", "tvt_code"]
meta_cols = ["id", "label", "tvt_code"]
ls_features = [cname for cname in pdf_features_label.columns if cname not in meta_cols]

#
print("Number of features: {}".format(len(ls_features)))
print("pdf_features_label: {}".format(pdf_features_label.shape))
print("pdf_train: {}".format(pdf_train.shape))
# read model
#with open("/home/stack/Data_science /data/models/xgb_model_{}.mod".format(version), "rb") as input_file:
with open("xgb_model_{}.mod".format(version), "rb") as input_file:
    res_model = pickle.load(input_file)
print(res_model.keys())
# load most important feature set
pdf_imp = pd.DataFrame(res_model["imp"])
pdf_imp.rename(columns= {0: "feat_name", 1: "F-score"}, inplace=True)
pdf_imp.set_index("feat_name", inplace=True)
print(pdf_imp.head(5))
# analyse tracking important features
ls_imp = []
for track in res_model["ls_tracked_imp"]:
    imp = pd.DataFrame(track)
    imp.rename(columns= {0: "feat_name", 1: "imp"}, inplace=True)
    ls_imp.append(imp)
pdf_analysis = pd.concat(ls_imp).groupby("feat_name").agg({"imp": ["max", "min", "mean", "std"], "feat_name": "count"})

# rename columns
name01 = pdf_analysis.columns.get_level_values(0)
name02 = pdf_analysis.columns.get_level_values(1)
rename_cols = ["{}_{}".format(tpl[0], tpl[1]) for tpl in zip(name01, name02)]
rename_cols[-1] = "num_chosen_by_model"
pdf_analysis.columns = rename_cols
pdf_analysis = pdf_analysis.sort_values(by="imp_max", ascending=False)
print(pdf_analysis.head())
print(pdf_analysis.tail())


def plot_distribution_over_target(pdf_input, cname):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    #
    #pdf_input.groupby("TARGET")[cname].plot(kind='kde', ax=axes[0], rot=45)
    #pdf_input.groupby("TARGET")[cname].hist(bins=100, ax=axes[1], xrot=45)
    #pdf_input.boxplot(column=cname, by='TARGET', ax=axes[2])

    pdf_input.groupby("label")[cname].plot(kind='kde', ax=axes[0], rot=45)
    pdf_input.groupby("label")[cname].hist(bins=100, ax=axes[1], xrot=45)
    pdf_input.boxplot(column=cname, by='label', ax=axes[2])

    #
    plt.suptitle("Distribution of {} (0: blue, 1: red)".format(cname))
    plt.show()


def my_auc(y_score, y_true, flexible_sign=True):
    # filter NaN
    idx = np.isfinite(y_score)
    xxx = y_score[idx]
    yyy = y_true[idx]

    # if label not only 1s/0s
    if yyy.std() > 0.0:
        auc = metrics.roc_auc_score(y_score=xxx, y_true=yyy)
    else:
        auc = 0.5

    # for evaluation only
    if (auc < 0.5) & (flexible_sign):
        auc = 1.0 - auc
    return auc


def feature_evaluate(pdf_feat_label, ls_feat=None):
    out_res = {
        "feat_name": [],
        "auc": [],
        "corr": [],
        "coverage": []
    }

    # calculate correlation
    #pdf_corr = pdf_feat_label[["TARGET"] + ls_feat].corr()
    pdf_corr = pdf_feat_label[["label"] + ls_feat].corr()

    for feat in ls_feat:
        out_res["feat_name"].append(feat)
        #out_res["auc"].append(my_auc(pdf_feat_label[feat], pdf_feat_label["TARGET"]))
        #out_res["corr"].append(pdf_corr.loc[feat, "TARGET"])
        out_res["auc"].append(my_auc(pdf_feat_label[feat], pdf_feat_label["label"]))
        out_res["corr"].append(pdf_corr.loc[feat, "label"])
        out_res["coverage"].append((~pdf_feat_label[feat].isna()).mean())

    #
    pdf_res = pd.DataFrame(out_res)
    pdf_res = pdf_res[["feat_name", "auc", "corr", "coverage"]].sort_values(by="auc", ascending=False)
    pdf_res.set_index("feat_name", inplace=True)
    return pdf_res
ls_feat = ["prev_app_NAME_PRODUCT_TYPE_walk_in_max", "baseline_extend_EXT_SOURCE_2", "credit_card_balance_AMT_DRAWINGS_CURRENT_min"]
pdf_eval01 = feature_evaluate(pdf_train, ls_feat=ls_feat)
pdf_eval01 = pdf_analysis.join(pdf_eval01, how="inner")
print(pdf_eval01)


