# The idea is to turn
# the results obtained with the committee
# into a summan.smr file, so that using
# the "TreesNoUnknown" label mode, we can
# filter out images considered `unknown'.
#
# Two summans will be created for `unknown' images
# misclassified as `known/clean'. In one the 
# `unknown' images will be assigned to the negative class
# and on the other they'll be assigned to the positive class.

import os
import argparse
import pandas as pd
import json


with open("/scratch/arturao/hyperelf/utils/dataset_paths.json", "r") as f:
    dataset_paths = json.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("report_path", metavar="report-path", type=str,
        help="Input file with the committee outputs.")
    parser.add_argument("--reference-summan",
        type=str, default="summanTrain",
        help="(Default: summanTrain) Which summan should be used as reference for known labels.")
    parser.add_argument("--dl-threshold", type=float, default=0,
        help="(Default 0) Disagreement Level (DL) threshold for considering images `unknown'. Any image with a strictly larger DL will be considered `unknown'.")
    parser.add_argument("--output-path", type=str, default="",
        help="(Optional) folder where to output the generated summan.smr")
    
    args = parser.parse_args()
    output_path = args.output_path
    dl_max = args.dl_threshold

    df_ref = pd.read_csv(dataset_paths[args.reference_summan])
    df_ref = df_ref.set_index("img_name")

    df = pd.read_csv(args.report_path)
    df = df.set_index('images')
    # Needed because in the reports the index (i.e. image names)
    # don't include the paths.
    df['imges_names'] = [os.path.basename(i) for i in df.index]
    df_unknown = df_ref.loc[list(df[df['disagreement'] > dl_max]['imges_names'])]
    df_easy = df_ref.loc[list(df[df['disagreement'] <= dl_max]['imges_names'])]

    df_unknown.to_csv(os.path.join(args.output_path, "summan_predicted_unknown.smr"))
    df_easy.to_csv(os.path.join(args.output_path, "summan_predicted_easy.smr"))
    with open(os.path.join(args.output_path, "info_rep_dis_csv2summan.txt"), "w") as f:
        f.write("Info about the experiment performed with the script report_disagreements_csv_to_summan.py\n\n")
        f.write(f"report_path: {args.report_path}\n")
        f.write(f"--reference-summan: {args.reference_summan}\n")
        f.write(f"--dl-threshold: {args.dl_threshold}\n")
        f.write(f"--output-path: {args.output_path}\n")




    
