# This script is used to combine the
# results on the DL predictions for two
# ensembles e1 and e2.
#
# Let d1(i) and d2(i) be the DL computed
# by e1 and e2 for an image i, respectivelly.
#
# In this script we define the combined
# DL 'c' as: c(d1, d2)(i) := max(d1(i), d2(i))
#
# It is assumed that the first report file
# will have the "training" images followed
# by the "validation" ones, and the second report
# file will have the opposite order.

import os
import json
import argparse
import numpy as np
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("report_path1", metavar="report-path1", type=str,
        help="Input file with the first committee outputs.")
    parser.add_argument("report_path2", metavar="report-path2", type=str,
        help="Input file with the second committee outputs.")
    parser.add_argument("index_file_train", metavar="index-file-train", type=str,
        help="Input file with the train indexes.")
    parser.add_argument("index_file_valid", metavar="index-file-valid", type=str,
        help="Input file with the validation indexes.")
    parser.add_argument("--output-filepath", type=str, default="",
        help="(Optional) filepath where to output the combined predictions.")
    
    args = parser.parse_args()
    output_filepath = args.output_filepath
    if output_filepath is None:
        output_filepath = "report_merged.csv"

    train_indexes = np.load(args.index_file_train)
    valid_indexes = np.load(args.index_file_valid)

    df1_index = np.concatenate([train_indexes,valid_indexes])
    df2_index = np.concatenate([valid_indexes,train_indexes])

    df1 = pd.read_csv(args.report_path1)
    df1 = df1.set_index('images')
    df1.index = df1_index
    df1.index.name = 'images'
    df1 = df1.sort_index()

    df2 = pd.read_csv(args.report_path2)
    df2 = df2.set_index('images')
    df2.index = df2_index
    df2.index.name = 'images'
    df2 = df2.sort_index()

    dfc = pd.DataFrame(index=df1.index)
    dfc['disagreement'] = np.max([df1['disagreement'].values, df2['disagreement'].values], axis=0)
    dfc['agreement_complement'] = np.max([df1['agreement_complement'].values, df2['agreement_complement'].values], axis=0)

    dfc.to_csv(output_filepath)






    
