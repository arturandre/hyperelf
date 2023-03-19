# compile all the 'results.csv' files from a single
# experiment in a single 'results_{expname}.csv'
# E.g. results.csv from all early exits and the final exit
# are put together in a single file named after the experiment's name.
# The compiled file should be ordered by the name of the images.
# It should contain the new field exp_name:
# image_name, exp_name, entropy, gt, pred, correct

# Then each experiment will be a column on the graph
# of comparisons. 

import os
import json
import shutil
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from shutil import rmtree
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare experiments defined in a configuration json file.")
    # dash-to-underscode doesn't work for positional arguments.
    # This can be fixed using the metavar argument.
    # ref: https://stackoverflow.com/a/20250435/3562468
    parser.add_argument('config_json', metavar='config-json', type=str,
        help="Configuration file with paths and names of the experiments to be compared.")
    parser.add_argument('output_folder', metavar='output-folder', type=str,
        help="Where the output files should be stored.")
    parser.add_argument('output_csv', metavar='output-csv', type=str,
        help=("""Name of the output csv with the aggregated information from the experiments.
        This is usefull when using the same output-folder for multiple sets of experiments."""))
    parser.add_argument('images_folder', metavar='images-folder', type=str, default=None,
        help=("""Output sub-folder with the compared images organized by voting counts.
        This sub-folder will be created under the folder defined by the 'output-folder' argument."""))
    parser.add_argument('--copy-images', action='store_true',
        help=("""This argument is optional and by setting it all the images compared
        will be copied to the subfolders defined by the 'images-folder' argument."""))
    # read experiments from a file
    # /scratch/arturao/hyperelf/outputs/exp027tu/it
    args = parser.parse_args()
    config_json = args.config_json
    with open(config_json, 'r') as f:
        config = json.load(f)
        exp_folders = config['paths']
        experiment_names = config['names']

    output_folder = args.output_folder
    #output_folder = "/scratch/arturao/hyperelf/outputs/combined"
    output_csv = args.output_csv
    #output_csv = "all_preds_t.csv"
    images_folder = args.images_folder
    #images_folder = "images_t"
    output_images = os.path.join(output_folder, images_folder)
    graphs_folder = os.path.join(output_folder, f'{images_folder}_graphs')
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(graphs_folder, exist_ok=True)

    # exp_folders = [
    #     "/scratch/arturao/hyperelf/outputs/exp024t/it",
    #     "/scratch/arturao/hyperelf/outputs/exp025t/it",
    #     "/scratch/arturao/hyperelf/outputs/exp026t/it",
    #     "/scratch/arturao/hyperelf/outputs/exp027t/it",
    #     "/scratch/arturao/hyperelf/outputs/exp028t/it",
    #     ]
    # experiment_names = [
    #     "exp024t",
    #     "exp025t",
    #     "exp026t",
    #     "exp027t",
    #     "exp028t"
    #     ]
    df_concat = None
    for exp_folder, exp_name in \
        (pbar := tqdm(zip(exp_folders, experiment_names))):
        pbar.set_description(f"Processing exp: {exp_name}")

        report_csv = None
        out_file = os.path.join(output_folder, f"{exp_name}.csv")
        for root, dirs, files in os.walk(exp_folder):
            for f in files:
                if f == "report.csv":
                    if report_csv is None:
                        report_csv = pd.read_csv(os.path.join(root, f), index_col=0)
                        report_csv.columns = report_csv.columns.str.strip() 
                    else:
                        aux_csv = pd.read_csv(os.path.join(root, f), index_col=0)
                        aux_csv.columns = aux_csv.columns.str.strip() 
                        report_csv = pd.concat([report_csv, aux_csv])
        report_csv.to_csv(out_file)
        if df_concat is None:
            df_concat = pd.DataFrame(index=report_csv.index)
            df_concat['gt'] = report_csv['gt']
        df_concat.insert(0, exp_name, report_csv['pred'])
        df_concat[f'entropy_{exp_name}'] = report_csv['entropy']
        df_concat[f'last exit_{exp_name}'] = report_csv['last exit']

    df_concat.sort_values(experiment_names, inplace=True)
    df_concat["Agreements - 0"] = (df_concat[experiment_names[0]] == 0).astype(int)
    df_concat["Agreements - 1"] = (df_concat[experiment_names[0]] == 1).astype(int)
    for exp_name in experiment_names[1:]:
        df_concat["Agreements - 0"] += (df_concat[exp_name]==0).astype(int)
        df_concat["Agreements - 1"] += (df_concat[exp_name]==1).astype(int)
    df_concat.to_csv(os.path.join(output_folder, output_csv))
    
    num_classifiers = len(experiment_names)

    if os.path.exists(output_images):
        rmtree(output_images)
    correct_folder = os.path.join(output_images, "correct")
    incorrect_folder = os.path.join(output_images, "incorrect")
    correct_df = \
        (
            ((df_concat["Agreements - 0"] >
            df_concat["Agreements - 1"]) & # simulating an 'and'
            (df_concat["gt"] == 0)) | # simulating an 'or'
            ((df_concat["Agreements - 1"] >
            df_concat["Agreements - 0"]) & # simulating an 'and'
            (df_concat["gt"] == 1))
        )
    incorrect_df = df_concat[~correct_df]
    correct_df = df_concat[correct_df]
    if len(incorrect_df) + len(correct_df) != len(df_concat):
        raise Exception("Correct and incorrect cases don't sum up to total. Possibly there are ties!")
    #incorrect_df = pd.concat([df_concat, correct_df]).drop_duplicates(keep=False)
    vote_groups = []
    entropy_groups = []
    stderr_groups = []
    exit_groups = []
    for i in range(num_classifiers+1):
        aux_correct = correct_df[correct_df["Agreements - 0"] == i]
        aux_incorrect = incorrect_df[incorrect_df["Agreements - 0"] == i]
        aux_correct_folder = os.path.join(correct_folder, f"{i}_{num_classifiers-i}")
        aux_incorrect_folder = os.path.join(incorrect_folder, f"{i}_{num_classifiers-i}")
        os.makedirs(aux_correct_folder)
        os.makedirs(aux_incorrect_folder)
        aux_correct.to_csv(os.path.join(aux_correct_folder, "correct.csv"))
        aux_incorrect.to_csv(os.path.join(aux_incorrect_folder, "incorrect.csv"))
        if args.copy_images:
            for j in aux_correct.index:
                shutil.copy(j, aux_correct_folder)
            for j in aux_incorrect.index:
                shutil.copy(j, aux_incorrect_folder)
        # Histogram on votes
        size_vote_group = (df_concat["Agreements - 0"] == i).sum()
        vote_groups.append(
            (
                f"{i}-{num_classifiers-i}",
                size_vote_group
            )
        )
        # Histogram on entropy averages per voting group
        total_num = 0
        total_sum = 0
        total_sum2 = 0
        last_exit_normalized_sum = 0
        for exp_name in experiment_names:
            aux = df_concat[df_concat["Agreements - 0"] == i][f"entropy_{exp_name}"]
            aux_exit_max = df_concat[f'last exit_{exp_name}'].max()
            aux_exit = df_concat[df_concat["Agreements - 0"] == i][f'last exit_{exp_name}']
            total_num += len(aux)
            total_sum += aux.sum()
            total_sum2 += (aux**2).sum()
            last_exit_normalized_sum += (aux_exit/aux_exit_max).sum()
        total_diff2 = 0
        entaverage = total_sum/total_num
        exitaverage = last_exit_normalized_sum/total_num
        for exp_name in experiment_names:
            aux = df_concat[df_concat["Agreements - 0"] == i][f"entropy_{exp_name}"]
            total_diff2 += ((aux-entaverage)**2).sum()
        # total_diff2 = variance
        entstderr = np.sqrt(total_diff2/total_num)
        entropy_groups.append(
            (
                f"{i}-{num_classifiers-i}",
                entaverage
            )
        )
        stderr_groups.append(
            (
                f"{i}-{num_classifiers-i}",
                #Ref: https://en.wikipedia.org/wiki/Standard_deviation#Definition_of_population_values
                #np.sqrt(total_sum2/total_num-(total_sum/total_num)**2)
                entstderr
            )
        )
        exit_groups.append(
            (
                f"{i}-{num_classifiers-i}",
                exitaverage
            )
        )
        
    aux_vote = {}
    aux_entropy = {}
    aux_stderr = {}
    aux_exit = {}
    midpoint = int((num_classifiers+1)/2)
    indices = list(range(num_classifiers+1))
    for i in indices[midpoint:]+indices[:midpoint]:
        k,v = vote_groups[i]
        aux_vote[k] = v
        k,v = entropy_groups[i]
        aux_entropy[k] = v
        k,v = stderr_groups[i]
        aux_stderr[k] = v
        k,v = exit_groups[i]
        aux_exit[k] = v

    vote_groups = aux_vote
    entropy_groups = aux_entropy
    stderr_groups = aux_stderr
    stderr_groups = aux_stderr
    exit_groups = aux_exit

    fig, ax = plt.subplots()
    ax.bar(range(len(vote_groups.values())), vote_groups.values())
    ax.set_xticks(range(len(vote_groups.values())))
    ax.set_xticklabels(labels=vote_groups.keys(), rotation='vertical')
    ax.set_title("Histogram for votes negative-positive.")
    ax.set_ylabel("# Images")
    ax.set_xlabel("Voting group")
    fig.savefig(os.path.join(graphs_folder, f'hist_votes.png'))
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.bar(range(len(entropy_groups.values())),
        entropy_groups.values(),
        yerr=list(stderr_groups.values()),
        ecolor='red')
    ax.set_title("Histogram for average entropy per voting group.")
    ax.set_ylabel("Entropy")
    ax.set_xlabel("Voting group")
    ax.set_xticks(range(len(entropy_groups.values())))
    ax.set_xticklabels(labels=entropy_groups.keys(), rotation='vertical')
    fig.savefig(os.path.join(graphs_folder, f'hist_avg_entropy.png'))
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.bar(range(len(exit_groups.values())), exit_groups.values())
    ax.set_xticks(range(len(exit_groups.values())))
    ax.set_xticklabels(labels=exit_groups.keys(), rotation='vertical')
    ax.set_title("Histogram for normalized early exits.")
    ax.set_ylabel("Normalized early exit")
    ax.set_xlabel("Voting group")
    fig.savefig(os.path.join(graphs_folder, f'hist_exits.png'))
    plt.close(fig)

    print(f"Done! Results at: '{output_folder}'.")
        

        