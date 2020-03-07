import os, math, subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# some settings for displaying Pandas results
pd.set_option('display.width', 2000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.precision', 4)
pd.set_option('display.max_colwidth', -1)

ls_files = pd.read_csv('train.csv')

def exploring_stats(pdf_input):
    # check rows, cols
    total_records = pdf_input.shape[0]
    total_columns = pdf_input.shape[1]
    print("Total records:", total_records)
    print("Total columns:", total_columns)

    # check dtypes
    name = []
    sub_type = []
    for n, t in pdf_input.dtypes.iteritems():
        name.append(n)
        sub_type.append(t)

    # check distinct
    ls_ndist = []
    for cname in pdf_input.columns:
        ndist = pdf_input[cname].nunique()
        pct_dist = ndist * 100.0 / total_records
        ls_ndist.append("{} ({:0.2f}%)".format(ndist, pct_dist))

    # check missing
    ls_nmiss = []
    for cname in pdf_input.columns:
        nmiss = pdf_input[cname].isnull().sum()
        pct_miss = nmiss * 100.0 / total_records
        ls_nmiss.append("{} ({:0.2f}%)".format(nmiss, pct_miss))

    # check zeros
    ls_zeros = []
    for cname in pdf_input.columns:
        try:
            nzeros = (pdf_input[cname] == 0).sum()
            pct_zeros = nzeros * 100.0 / total_records
            ls_zeros.append("{} ({:0.2f}%)".format(nzeros, pct_zeros))
        except:
            ls_zeros.append("{} ({:0.2f}%)".format(0, 0))
            continue

    # check negative
    ls_neg = []
    for cname in pdf_input.columns:
        try:
            nneg = (pdf_input[cname].astype("float") < 0).sum()
            pct_neg = nneg * 100.0 / total_records
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
    pdf_stats = pdf_input.describe().transpose()
    ls_stats = []
    for stat in pdf_stats.columns:
        data[stat] = []
        for cname in pdf_input.columns:
            try:
                data[stat].append(pdf_stats.loc[cname, stat])
            except:
                data[stat].append(0.0)

    # take samples
    nsample = 10
    pdf_sample = pdf_input.sample(frac=.5).head(nsample).transpose()
    pdf_sample.columns = ["sample_{}".format(i) for i in range(nsample)]

    # output
    col_ordered = ["sub_type", "n_distinct", "n_miss", "n_negative", "n_zeros",
                   "25%", "50%", "75%", "count", "max", "mean", "min", "std"] + list(pdf_sample.columns)
    pdf_data = pd.DataFrame(data).set_index("name")
    pdf_data = pd.concat([pdf_data, pdf_sample], axis=1)
    pdf_data = pdf_data[col_ordered]

    return pdf_data
ls_report = {}
for f in ls_files:
    print("Exploring {}".format(f))
    ls_report[f] = exploring_stats(ls_files)
    ls_report[f].to_csv('simple_submission.csv', index=False)
    break