import os, pickle
import pandas as pd
import numpy as np
#
import matplotlib.pyplot as plt
from IPython.display import display
#
from sklearn import metrics
from sklearn.model_selection import train_test_split
#
import xgboost as xgb
from xgboost import plot_importance

# some settings for displaying Pandas results
pd.set_option('display.width', 2000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.precision', 4)
version = "v07"
# specified features set for joining
ls_feat_file = [
    'baseline.pkl.bz2',
    'baseline_extend.pkl.bz2',
]

# use first features for base joined
feat_path = ls_feat_file[0]#os.path.join("/home/stack/Data_science /data/", ls_feat_file[0])
pdf_combined = pd.read_pickle(feat_path, compression="bz2")
# join next features set
for fname in ls_feat_file[1:]:
    feat_path = fname#os.path.join("/home/stack/Data_science /data/", fname)
    pdf_feat = pd.read_pickle(feat_path, compression="bz2")
    print(fname, pdf_feat.shape)

    # add table prefix
    tbl_prefix = fname.split(".")[0]
    #rename_col = {cname: "{}_{}".format(tbl_prefix, cname) for cname in pdf_feat.columns if cname != "SK_ID_CURR"}
    rename_col = {cname: "{}_{}".format(tbl_prefix, cname) for cname in pdf_feat.columns if cname != "id"}
    pdf_feat.rename(columns=rename_col, inplace=True)

    # join
    #pdf_combined = pdf_combined.merge(pdf_feat, on="SK_ID_CURR", how="left")
    pdf_combined = pdf_combined.merge(pdf_feat, on="id", how="left")

print("rows, columns", pdf_combined.shape)
#ls_features = [feat for feat in pdf_combined.columns if feat not in ["SK_ID_CURR"]]
ls_features = [feat for feat in pdf_combined.columns if feat not in ["id"]]

# join with label
pdf_tvt = pd.read_pickle("pdf_tvt_extend.pkl", compression="bz2")
#pdf_features_label = pdf_tvt.merge(pdf_combined, on="SK_ID_CURR", how="left")
pdf_features_label = pdf_tvt.merge(pdf_combined, on="id", how="left")
print(pdf_features_label.shape)


# # read model baseline
# with open("/home/stack/Data_science /data/models/xgb_model_baseline_{}.mod".format(version), "rb") as input_file:
#     res_model = pickle.load(input_file)
# print(res_model.keys())
# read model
with open("xgb_model_{}.mod".format(version), "rb") as input_file:
    res_model = pickle.load(input_file)
print(res_model.keys())

#meta_cols = ["SK_ID_CURR", "TARGET", "tvt_code"]
meta_cols = ["id", "label", "tvt_code"]
pdf_features_label = pd.read_csv("pdf_features_label.csv.bz2", compression="bz2")

from sklearn.model_selection import GridSearchCV, StratifiedKFold
pdf_data = pdf_features_label[pdf_features_label["tvt_code"].isin(["train", "val", "test"])].copy()
pdf_data.shape

param_grid = {
    "objective": ["binary:logistic"],
    "booster": ["gbtree"],
    "max_depth": [4, 7], # default: 3 only for depthwise
    "n_estimators": [1000], # default: 500
    "learning_rate": [0.025], # default: 0.05
    "subsample": [0.6, 0.8],
    "colsample_bytree": [0.6, 0.8],  # default:  1.0
    "colsample_bylevel": [0.6, 0.8], # default: 1.0
    "random_state": [1],
    'min_child_weight': [11],

    #
    "silent": [True],
    'seed': [1]
}

xgb_model = xgb.XGBClassifier()
grid_search = GridSearchCV(xgb_model, param_grid, n_jobs=16,
                   cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=1),
                   scoring='roc_auc',
                   verbose=2)

#grid_result = grid_search.fit(pdf_data[ls_features], pdf_data["TARGET"])
grid_result = grid_search.fit(pdf_data[ls_features], pdf_data["label"])

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

#
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
X_kaggle_test = pdf_features_label.query("tvt_code == 'kahapa'")[ls_features]
y_test_pred = grid_search.predict_proba(X_kaggle_test)[:, 1]


#SK_IDs = pdf_features_label.query("tvt_code == 'kahapa'")["SK_ID_CURR"].tolist()
SK_IDs = pdf_features_label.query("tvt_code == 'kahapa'")["label"].tolist()
#pdf_submiss = pd.DataFrame({"SK_ID_CURR": SK_IDs, "TARGET": y_test_pred})
pdf_submiss = pd.DataFrame({"id": SK_IDs, "label": y_test_pred})
pdf_submiss.to_csv("submission_gridsearch_{}.csv".format(version), index=True)
pdf_submiss.head()






