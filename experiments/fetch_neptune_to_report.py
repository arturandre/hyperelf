from argparse import ArgumentParser
import neptune.new as neptune
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def get_experiment_data(neptune_exp_id):
    #with_id="HYPER-457")
    run = neptune.init_run(
        project="inacity/HyperElyx",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2NmZmOTAzYi02NTJkLTQ0MzUtOTYzYi1kYjVjZTVjYzc4MmMifQ==",
        with_id=neptune_exp_id)

    # Fetch highest accuracy so far
    label_mode = run["parameters/dataset"].fetch()
    if label_mode in ["TreesUnknownPositive", "TreesCustomUnknownPositive"]:
        label_mode = "positive"
    elif label_mode in ["TreesUnknownNegative", "TreesCustomUnknownNegative"]:
        label_mode = "negative"
    elif label_mode in ["TreesNoUnknown", "TreesCustomNoUnknown", "TreesHalfNoUnknown"]:
        label_mode = "remove"

    model_name, elyx = run["parameters/model"].fetch().split("Elyx")
    if model_name == "MobileNetV3Large":
        model_name = "MobileNetV3 Large"
    elyx = "No" if elyx == "None" else "Yes"

    test_acc = run["test/correct"].fetch_values()
    train_acc = run["train/correct"].fetch_values()

    test_max_idx = test_acc.loc[:, "value"].idxmax()
    test_max_acc = test_acc.loc[test_max_idx, "value"]
    train_maxtest_acc = train_acc.loc[test_max_idx, "value"]
    ret = {
        "model_name": model_name,
        "label_mode": label_mode,
        "elyx": elyx,
        "train_maxtest_acc": train_maxtest_acc,
        "test_max_acc": test_max_acc
    }
    return ret

def exp_data_to_latex_tabular_row(exp_data, bold_test=False):
    if exp_data is dict:
        model_name, label_mode, elyx, train_maxtest_acc, test_max_acc = exp_data.values()
    else:
        model_name, label_mode, elyx, train_maxtest_acc, test_max_acc = exp_data
    aux = "100\\% & " if train_maxtest_acc == 1 else f"{100*train_maxtest_acc:.2f}\\% & "
    train_maxtest_acc = aux
    aux = "100\\%" if train_maxtest_acc == 1 else f"{100*test_max_acc:.2f}\\%"
    test_max_acc = aux
    if bold_test:
        test_max_acc = f"\\textbf{{{test_max_acc}}}"
    test_max_acc += " \\\\"
    tab_row = (
        f"{model_name} & "
        f"{label_mode} & "
        f"{elyx} & "
        f"{train_maxtest_acc}"
        f"{test_max_acc}"
    )
    return tab_row

def exp_dataframe_to_graphs(exp_df, suffix="", output_folder=""):
    # Removed vs Positive vs Negative
    # No early exit
    # Across architectures
    markers = ["o", "v", "1", "s", "+", "x", "D", "3", "^", "*"]
    #groups = exp_df.groupby(["model_name", "elyx", "label_mode"])
    groups_no_elyx = exp_df[exp_df['elyx'] == "No"].groupby("label_mode")
    groups_remove = exp_df[exp_df['label_mode'] == "remove"].groupby("elyx")
    # for group in groups:
    #     if group[1] == "No":
    #         groups_no_elyx.append(groups.get_group(group))
    #     if group[2] == "remove":
    #         groups_remove.append(groups.get_group(group))

    # first = groups_no_elyx[0]
    # for group in groups_no_elyx[1:]:
    #     first = pd.concat([first, group])
    # groups_no_elyx = first

    # first = groups_remove[0]
    # for group in groups_remove[1:]:
    #     first = pd.concat([first, group])
    # groups_remove = first

    half_group_padding = 0.05
    group_width = 1.0
    num_groups = len(groups_no_elyx) #removed, positive, negative
    if num_groups > 0:
        fig, ax = plt.subplots()
        #num_groups = 3 #removed, positive, negative
        #group_names = ["removed", "positive", "negative"]
        x_centers = np.arange(num_groups)
    rects = []
    legend_printed = False
    for i, (name, group) in enumerate(groups_no_elyx):
        df = group
        nbars = len(df)
        bar_width = (group_width-half_group_padding*2.0)/(nbars)
        if not legend_printed:
            legend_printed = True
            labels = label=df['model_name'].values
        else:
            labels = None
        rect = ax.bar(
            x_centers[i]+\
            (np.arange(nbars)*(bar_width))+\
            half_group_padding - ((group_width-bar_width)/2),
            np.around(df.loc[:, "test_max_acc"].values*100, 2),
            width=bar_width,
            label=labels,
            color=plt.get_cmap('tab20').colors[:len(df)]
        )
        rects.append(rect)
    ax.set_xticks(x_centers, groups_no_elyx.groups.keys())
    ax.set_yticks(list(range(0,101,20)))
    ax.legend(loc="lower right")
    ax.set_xlabel("Unknown label is:")
    ax.set_ylabel("Test accuracy (%)")

    for rect in rects:
        ax.bar_label(rect, padding=3, rotation="vertical")
    ax.set_ylim((0,110))
    fig.tight_layout()
    plt.savefig(os.path.join(output_folder, f"report_noelyx{suffix}.png"))
    plt.close()
    ###
    # Early exit vs Non-Early exit
    # Unknowns removed
    # Across architectures
    num_groups = len(groups_remove) #Yes, No
    if num_groups > 0:
        fig, ax = plt.subplots()
        x_centers = np.arange(num_groups)
    rects = []
    legend_printed = False
    for i, (name, group) in enumerate(groups_remove):
        df = group
        nbars = len(df)
        bar_width = (group_width-half_group_padding*2.0)/(nbars)
        if not legend_printed:
            legend_printed = True
            labels = label=df['model_name'].values
        else:
            labels = None
        rect = ax.bar(
            x_centers[i]+\
            (np.arange(nbars)*(bar_width))+\
            half_group_padding - ((group_width-bar_width)/2),
            np.around(df.loc[:, "test_max_acc"].values*100, 2),
            width=bar_width,
            label=labels,
            color=plt.get_cmap('tab20').colors[:len(df)]
        )
        rects.append(rect)
    ax.set_xticks(x_centers, groups_remove.groups.keys())
    ax.set_yticks(list(range(0,101,20)))
    ax.legend(loc="lower right")
    ax.set_xlabel("Has early exit regularization?")
    ax.set_ylabel("Test accuracy (%)")

    for rect in rects:
        ax.bar_label(rect, padding=3, rotation="vertical")
    ax.set_ylim((0,110))
    fig.tight_layout()
    plt.savefig(f"report_onlyelyx{suffix}.png")
    plt.close()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('prefix', default="", help="Experiment prefix, example: HYPER-")
    parser.add_argument('intervals', default=[], 
        help="Comma separated sequence of intervals, for example: 2-4,6-8,17,21 will include the experiments 2,3,4,6,7,8,17,21.")
    parser.add_argument('--suffix', help="(optional) Suffix for the output files.")
    parser.add_argument('--output-folder', help="(optional) Folder to save the output files.")
    args = parser.parse_args()
    intervals = args.intervals.split(",")
    suffix = "" if args.suffix is None else f"_{args.suffix}"
    output_folder = args.output_folder
    if output_folder is not None:
        os.makedirs(output_folder)
    else:
        output_folder = ""
    df = pd.DataFrame()
    for interval in intervals:
        if "-" in interval:
            l, h = interval.split("-")
            l = int(l)
            h = int(h)
        else:
            l = int(interval)
            h = l
        for i in range(l, h+1):
            #exp_data = get_experiment_data(f"{HYPER-}457")
            exp_data = get_experiment_data(f"{args.prefix}{i}")
            df_dict = pd.DataFrame([exp_data])
            df = pd.concat([df, df_dict], ignore_index=True)
    with open(os.path.join(output_folder, f"report{suffix}.csv"), "w") as report:
        df = df.sort_values(["model_name", "label_mode", "elyx"], ascending=False)
        exp_dataframe_to_graphs(df, suffix=suffix, output_folder=output_folder)
        max_test_idx = df.loc[:, "test_max_acc"].idxmax()
        for idx in df.index:
            tab_row = exp_data_to_latex_tabular_row(df.loc[idx, :].values, bold_test=idx==max_test_idx)
            print(tab_row)
            report.write(tab_row + "\n")

