import os, math, subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from IPython.display import display

# some settings for displaying Pandas results
pd.set_option('display.width', 2000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.precision', 4)
pd.set_option('display.max_colwidth', -1)


# train
pdf_train = pd.read_csv("train.csv", encoding='unicode_escape')#pd.read_csv('/home/stack/Data_science /data/train.csv', encoding= 'unicode_escape')
print("(rows, columns)", pdf_train.shape)
# test
pdf_test = pd.read_csv("test.csv", encoding='unicode_escape')#pd.read_csv('/home/stack/Data_science /data/test.csv', encoding= 'unicode_escape')
print("(rows, columns)", pdf_test.shape)

# check overlap
#train_ids = pdf_train["SK_ID_CURR"]
#test_ids = pdf_test["SK_ID_CURR"]
train_ids = pdf_train["id"]
test_ids = pdf_test["id"]
n_overlap = len(set(train_ids) & set(test_ids))
print("Number of overlap ids: {}".format(n_overlap))

# check duplicate
if len(train_ids.drop_duplicates()) == pdf_train.shape[0]:
    print("Train data no duplicate")
if len(test_ids.drop_duplicates()) == pdf_test.shape[0]:
    print("Test data no duplicate")

# get ids and target label
#pdf_tvt = pdf_train[["SK_ID_CURR", "TARGET"]].copy()
pdf_tvt = pdf_train[["id", "label"]].copy()
pdf_tvt["tvt_code"] = "kahapa"

# target
#y = pdf_tvt["TARGET"]
y = pdf_tvt["label"]
# index
#X = pdf_tvt["SK_ID_CURR"]
X = pdf_tvt["id"]
# train, val, test set will be 70%, 15%, 15% of the dataset respectively.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.17, random_state=1)

# check validity
assert X.shape[0] == X_train.shape[0] + X_val.shape[0] + X_test.shape[0], "TVT split not correct"

# assign code
pdf_tvt.loc[X_train.index, "tvt_code"] = "train"
pdf_tvt.loc[X_val.index, "tvt_code"] = "val"
pdf_tvt.loc[X_test.index, "tvt_code"] = "test"

# extend for kaggle test
#pdf_test["TARGET"] = -1
pdf_test["label"] = -1
pdf_test["tvt_code"] = "kahapa"

#pdf_tvt_extend = pd.concat([pdf_tvt, pdf_test[["SK_ID_CURR", "TARGET", "tvt_code"]]])
pdf_tvt_extend = pd.concat([pdf_tvt, pdf_test[["id", "label", "tvt_code"]]])
pdf_tvt_extend = pdf_tvt_extend.reset_index(drop=True)
pdf_tvt_extend.head()

# check validity
#assert pdf_tvt_extend.shape[0] == pdf_tvt_extend["SK_ID_CURR"].nunique(), "Extend TVT overlaped"
print(pdf_tvt_extend.shape[0])
print(pdf_tvt_extend["id"].nunique())
#assert pdf_tvt_extend.shape[0] == pdf_tvt_extend["id"].nunique(), "Extend TVT overlaped"

# check tvt percentage
pdf_tvt_check = pdf_tvt_extend["tvt_code"].value_counts().to_frame("cnt")
pdf_tvt_check["percentage"] = pdf_tvt_check["cnt"] * 100.0 / pdf_tvt.shape[0]
print(pdf_tvt_check)

pdf_tvt_extend.to_pickle("pdf_tvt_extend.pkl", compression="bz2")