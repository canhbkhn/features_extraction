import math, os, subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from common import calc_woe_and_iv

# some settings for displaying Pandas results
pd.set_option('display.width', 2000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.precision', 4)
pd.set_option('display.max_colwidth', -1)

# call api
data_path = 'train.csv'

# exploring data
# check cols, rows
def exploring_data(df_input):
    total_rows = df_input.shape[0]
    print("Total rows: ", total_rows)
    total_cols = df_input.shape[1]
    print("Total cols: ", total_cols)

# check data types
    name = []
    sub_type = []
    for n, t in df_input.dtypes.iteritems():
        name.append(n)
        #print(n)
        sub_type.append(t)

    # check distinct
    ls_ndist = []
    for cname in df_input.columns:
        ndist = df_input[cname].nunique()
        pct_dist = ndist * 100.0 / total_rows
        ls_ndist.append("{} ({:0.2f}%)".format(ndist, pct_dist))

    # check missing
    ls_nmiss = []
    for cname in df_input.columns:
        nmiss = df_input[cname].isnull().sum()
        pct_miss = nmiss * 100.0 / total_rows
        ls_nmiss.append("{} ({:0.2f}%)".format(nmiss, pct_miss))

    # check zeros
    ls_zeros = []
    for cname in df_input.columns:
        try:
            nzeros = (df_input[cname] == 0).sum()
            pct_zeros = nzeros * 100.0 / total_rows
            ls_zeros.append("{} ({:0.2f}%)".format(nzeros, pct_zeros))
        except:
            ls_zeros.append("{} ({:0.2f}%)".format(0, 0))
            continue

    # check negative
    ls_neg = []
    for cname in df_input.columns:
        try:
            nneg = (df_input[cname].astype("float") < 0).sum()
            pct_neg = nneg * 100.0 / total_rows
            ls_neg.append("{} ({:0.2f}%)".format(nneg, pct_neg))
        except:
            ls_neg.append("{} ({:0.2f}%)".format(0, 0))
            continue

    # prepare output
    data = {
        "name": name,
        "sub_type": sub_type,
        "n_distinct": ls_ndist,
        "n_miss": ls_nmiss,
        "n_zeros": ls_zeros,
        "n_negative": ls_neg,
    }

    # check stats
    pdf_stats = df_input.describe().transpose()
    ls_stats = []
    for stat in pdf_stats.columns:
        data[stat] = []
        for cname in df_input.columns:
            try:
                data[stat].append(pdf_stats.loc[cname, stat])
            except:
                data[stat].append(0.0)

    # take samples
    nsample = 10
    pdf_sample = df_input.sample(frac=.5).head(nsample).transpose()
    pdf_sample.columns = ["sample_{}".format(i) for i in range(nsample)]

    # output
    col_ordered = ["sub_type", "n_distinct", "n_miss", "n_negative", "n_zeros",
                   "25%", "50%", "75%", "count", "max", "mean", "min", "std"] + list(pdf_sample.columns)
    pdf_data = pd.DataFrame(data).set_index("name")
    pdf_data = pd.concat([pdf_data, pdf_sample], axis=1)
    pdf_data = pdf_data[col_ordered]

    return pdf_data

data_path = 'train.csv'
pdf_data = pd.read_csv(data_path)
ls_report = exploring_data(pdf_data)
print(ls_report)
ls_report.to_csv('data_exploring.csv', index=True)