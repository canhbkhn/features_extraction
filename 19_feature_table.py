import os, math, subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from IPython.display import display
from lib_feature_engineering import *

# some settings for displaying Pandas results
pd.set_option('display.width', 2000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.precision', 4)
pd.set_option('display.max_colwidth', -1)

# train

pdf_train = pd.read_csv('train.csv', encoding= 'unicode_escape')

# test
pdf_test = pd.read_csv('test.csv', encoding= 'unicode_escape')

# load meta data
pdf_meta = pd.read_csv('report.csv', encoding= 'unicode_escape')

#fillna of SEX
pdf_train['FIELD_8'].fillna(value='NOSEX',inplace=True)
pdf_test['FIELD_8'].fillna(value='NOSEX',inplace=True)
#fillna of province
pdf_train["province"]= pdf_train["province"].notna()
pdf_test["province"]= pdf_train["province"].notna()


#fillna of district
pdf_train["district"]= pdf_train["district"].notna()
pdf_test["district"]= pdf_train["district"].notna()

pdf_train["maCv"]= pdf_train["maCv"].notna()
pdf_test["maCv"]= pdf_train["maCv"].notna()

print(pdf_train["maCv"])
#fillna of field_23
pdf_train['FIELD_23'].fillna(value='FALSE',inplace=True)
pdf_test['FIELD_23'].fillna(value='FALSE',inplace=True)

ls_type_object = [
    "province", "district", "maCv", "FIELD_7", "FIELD_8", "FIELD_10", "FIELD_11", "FIELD_12", "FIELD_13", "FIELD_17",
    "FIELD_18", "FIELD_19", "FIELD_20", "FIELD_24", "FIELD_25", "FIELD_26", "FIELD_27", "FIELD_28",
    "FIELD_29", "FIELD_30", "FIELD_31", "FIELD_35", "FIELD_36", "FIELD_37", "FIELD_38", "FIELD_39", "FIELD_40",
    "FIELD_41", "FIELD_42", "FIELD_43", "FIELD_44", "FIELD_45",
]
for k in ls_type_object:
    pdf_train[k].fillna(value="NA", inplace=True)
    pdf_test[k].fillna(value="NA", inplace=True)

# filter by tvt code
pdf_tvt_extend = pd.read_pickle("pdf_tvt_extend.pkl", compression="bz2")
pdf_train_filtered = (pdf_tvt_extend.query("tvt_code == 'train'").merge(pdf_train[["id"]], on="id").drop(columns=["tvt_code"]))
                      #.merge(pdf_train[["SK_ID_CURR"]], on="SK_ID_CURR")
                      
pdf_train_filtered.head()


def gen_binary_one_hot_feat(pdf_input):
    pdf_data = pdf_input.copy()
    select_features = []
    dict_feat = {
        "binary_default": {
            "maCv": ['TRUE', 'FALSE'],
            "district": ['TRUE', 'FALSE'],
            "province": ['TRUE', 'FALSE'],
            "FIELD_8": ['NOSEX', 'FEMALE','MALE'],
            "FIELD_10": ['NA', 'T1', 'GH'],
            "FIELD_18": ['TRUE', 'FALSE', 'NA'],
            "FIELD_19": ['TRUE', 'FALSE', 'NA'],
            "FIELD_20": ['TRUE', 'FALSE', 'NA'],
            "FIELD_23": ['TRUE', 'FALSE'],
            "FIELD_25": ['TRUE', 'FALSE', 'NA'],
            "FIELD_26": ['TRUE', 'FALSE', 'NA'],
            "FIELD_27": ['TRUE', 'FALSE', 'NA'],
            "FIELD_28": ['TRUE', 'FALSE', 'NA'],
            "FIELD_29": ['TRUE', 'FALSE', 'None', 'NA'],
            "FIELD_30": ['TRUE', 'FALSE', 'None', 'NA'],
            "FIELD_31": ['None', 'FALSE', 'NA'],
            "FIELD_36": ['TRUE', 'FALSE', 'None', 'NA'],
            "FIELD_38": ['TRUE', 'FALSE', 'NA'],
            "FIELD_42": ['Zezo', 'One', 'None', 'NA'],
            "FIELD_44": ['Two', 'One', 'None', 'NA'],
            "FIELD_47": ['TRUE', 'FALSE'],
            "FIELD_48": ['TRUE', 'FALSE'],
            "FIELD_49": ['TRUE', 'FALSE'],

        },
        "binary": [
            "FIELD_1",
            "FIELD_14",
            "FIELD_15",
            "FIELD_32",
            "FIELD_33",
            "FIELD_34",
            "FIELD_46",
        ],
        "onehot": {
            "FIELD_9": ["na", "GD", "DN", "XD", "HC", "TN", "CH", "CN", "HT", "DT", "XK", "TK", "GB", "DK", "SV", "HN",
                        "TS", "TA", "HD", "NN", "BT", "HS", "HX", "NO", "KC", "TE", "CB", "TC", "XV", "80", "XN", "CC",
                        "86", "75", "79", "MS"],
            "FIELD_11": ["12", "3", "7", "0", "1", "6", "10", "24", "11", "60", "13", "8", "36", "2", "34", "5", "None",
                         "4", "9", "19", "15", "30", "20", "56", "25", "35", "69", "17", "54", "42", "14", "16", "21",
                         "22", "47", "26", "45", "28", "18", "27", "32", "37", "72", "70", "59","NA"],
            "FIELD_12": [ "None", "0", "1", "HT", "TN", 'NA'],
            "FIELD_13": ["BI", "YN", "TG", "TB", "QW", "AQ", "TO", "TI", "TC", "FI", "BO", "HI", "TZ", "HZ", "HW", "AC",
                         "HK", "TL", "BT", "TA", "TJ", "AO", "TD", "TN", "TH", "HB", "TK", "HF", "HJ", "KC", "DA", "TE",
                         "TM", "NT", "TX", "FF", "AI", "HD", "TP", "HG", "AR", "QA", "TY", "QZ", "TQ", "AP", "HH", "T5",
                         "HE", "HL", "TU", "TV", "FE", "TF", "HA", "HC", "TW", "TS", "T9", "QI", "BQ", "SN", "QB", "BD",
                         "BN", "0", "SG", "AB", "HX", "KT", "AL", "FJ", "DW", "BA", "SP", "YV", "BH", "TT", "NI", "FB",
                         "QD", "YA", "NK", "SY", "AA", "HY", "HR", "NE", "QG", "EB", "FA", "SL", "T7", "NQ", "BP", "CA",
                         "SS", "T3", "NW", "YF", "DE", "FV", "AD", "QF", "NC", "8", "QC", "SI", "HS", "SK", "QO", "FD",
                         "FK", "BK", "AH", "FL", "NV", "HP", "NB", "FH", "FC", "T1", "EI", "BE", "QE", "DC", "TR", "QU",
                         "A6", "FP", "FR", "SD", "BB", "QJ", "QL", "HQ", "FS", "AS", "CD", "HN", "EA", "CE", "H3", "EJ",
                         "CJ", "DQ", "DB", "DZ", "DT", "SA", "NO", "HO", "4", "NH", "SO", "QK", "F4", "SH", "HM", "EH",
                         "NU", "AY", "B2", "EP", "NG", "BG", "FM", "DH", "IA"],
            "FIELD_17": ["G8", "None", "G9", "G7", "GX", "G3", "G4", "G2", 'NA'],
            "FIELD_24": ["None", "K3", "K1", "K2", 'NA'],
            "FIELD_35": ["Zero", "Four", "One", "Three", "Two", 'NA'],
            "FIELD_37": ["TRUE", "FALSE", "None", 'NA'],
            "FIELD_39": ["VN", "None", "JP", "CA", "TQ", "CN", "UK", "CZ", "KR", "VU", "1", "TW", "IN", "DL", "HK", "US",
                         "KP", "DE", "NL", "NN", "DT", "HQ", "N", "SG", "MY", "IT", "BE", "TH", "DK", "TR", "IL", "TS",
                         "FR", "SE", "AU", "AE", "GB", "NU", "SC", "PH", "ES", "AD", "DM", "TL", "TK", 'NA'],
            "FIELD_40": ["None", "1", "2", "02 05 08 11", "3", "6", "4", "08 02", "05 08 11 02", 'NA'],
            "FIELD_41": ["V", "I", "III", "II", "IV", 'NA'],
            "FIELD_43": ["None", "B", "D", "C", "A", "5", "0", 'NA'],
            "FIELD_45": ["1", "2", "None", 'NA'],
        }
    }

    for k in dict_feat:
        if k == "binary_default":
            for cname in dict_feat[k]:
                # get default value
                default_val = dict_feat[k][cname][0]

                # convert category to binary
                feat_name = "is_" + cname
                select_features.append(feat_name)
                pdf_data[feat_name] = pdf_data[cname].apply(lambda x: int(x == default_val))
        elif k == "binary":
            # rename only
            for cname in dict_feat[k]:
                feat_name = "is_" + cname
                select_features.append(feat_name)
                pdf_data[feat_name] = pdf_data[cname]
        elif k == "onehot":
            for cname in dict_feat[k]:
                ls_vals = dict_feat[k][cname]
                for val in ls_vals:
                    try:
                        new_name = "{}_{}".format(cname, val.replace(" ", "_") \
                                                  .replace(":", "_") \
                                                  .replace("/", "_") \
                                                  .replace("-", "_"))

                        select_features.append(new_name)
                        pdf_data[new_name] = pdf_data[cname].apply(lambda x: int(x == val))
                    except Exception as err:
                        print("One hot for {}-{}. Error: {}".format(cname, val, err))

    #return pdf_data[["SK_ID_CURR"] + select_features]
    return pdf_data[["id"]+ select_features]

# for train feat
pdf01_baseline = gen_binary_one_hot_feat(pdf_train)

# for test feat
pdf02_baseline = gen_binary_one_hot_feat(pdf_test)

# print results
print(pdf01_baseline.shape, pdf02_baseline.shape)
display(pdf01_baseline.head().T)

eval_agg01 = feature_evaluate(pdf_train_filtered, pdf01_baseline)

sel_feat = eval_agg01.query("auc > 0.501")["name"].tolist()

# for train
#pdf01_baseline = pdf01_baseline[["SK_ID_CURR"] + sel_feat]
pdf01_baseline = pdf01_baseline[["id"] + sel_feat]

# for test
#pdf02_baseline = pdf02_baseline[["SK_ID_CURR"] + sel_feat]
pdf02_baseline = pdf02_baseline[["id"] + sel_feat]

s_dtype = pdf_train.dtypes
ls_continuous_name = s_dtype[s_dtype == "float64"].index.tolist()

# for train feat
#pdf11_baseline = pdf_train[["SK_ID_CURR"] + ls_continuous_name].copy()
pdf11_baseline = pdf_train[["id"] + ls_continuous_name].copy()

# for test feat
#pdf12_baseline = pdf_test[["SK_ID_CURR"] + ls_continuous_name].copy()
pdf12_baseline = pdf_test[["id"] + ls_continuous_name].copy()

def store_features(pdf_train, pdf_test, fname):
    print(pdf_train.shape, pdf_test.shape)
    fname = os.path.join("", "{}.pkl.bz2".format(fname))
    pdf_out = pd.concat([pdf_train, pdf_test]).reset_index(drop=True)
    pdf_out.to_pickle(fname, compression="bz2")
    print("Store features completed!")
store_features(pdf01_baseline, pdf02_baseline, "baseline")
store_features(pdf11_baseline, pdf12_baseline, "baseline_extend")
