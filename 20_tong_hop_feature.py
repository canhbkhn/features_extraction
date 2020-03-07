import os, subprocess, pickle
import pandas as pd
import numpy as np
from IPython.display import display
# from lib_feature_engineering import *
# specified features set for joining
ls_feat_file = [
    'baseline.pkl.bz2',
    'baseline_extend.pkl.bz2'
]
# use first features for base joined
#feat_path = os.path.join("/home/stack/Data_science /data/", ls_feat_file[0])
#pdf_combined = pd.read_pickle(feat_path, compression="bz2")
pdf_combined = pd.read_pickle("baseline.pkl.bz2", compression="bz2")

# join next features set
for fname in ls_feat_file[1:]:
    #feat_path = os.path.join("/home/stack/Data_science /data/", fname)
    pdf_feat = pd.read_pickle("baseline_extend.pkl.bz2", compression="bz2")#pd.read_pickle(feat_path, compression="bz2")
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
pdf_tvt = pd.read_pickle("pdf_tvt_extend.pkl", compression="bz2")
print(pdf_tvt["tvt_code"].value_counts())


#pdf_features_label = pdf_tvt.merge(pdf_combined, on="SK_ID_CURR", how="left")
pdf_features_label = pdf_tvt.merge(pdf_combined, on="id", how="left")
# print(pdf_features_label.shape)
# display(pdf_features_label.head().T)
pdf_features_label.to_csv("pdf_features_label.csv.bz2", compression="bz2")