# This script includes a new column in a report_disagreements.csv
# corresponding to the cscore of a sample from some cscore.npy file.
#
# Matching the indexes may require attention.

import shutil
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import os
import numpy as np
import pandas as pd
from tqdm import tqdm 
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from scipy.stats import gaussian_kde

import os, sys
sys.path.append( # "."
    os.path.dirname( #"experiments/"
    os.path.dirname( #"hyperelf/" 
        os.path.abspath(__file__))))

from utils.dataset import prepare_dataset

import matplotlib
matplotlib.use('Agg')

def plot_regline(x_1dim, y_1dim, ax, output_file=None):
    """
    - x, y and yhat shapes should be [batch, ...]
    - ax is a matplotlib axis
    """
    x = x_1dim.reshape(-1, 1)
    y = y_1dim.reshape(-1, 1)
    # Model initialization
    regression_model = LinearRegression()
    # Fit the data(train the model)
    regression_model.fit(x, y)
    # Predict
    yhat = regression_model.predict(x)

    # model evaluation
    rmse = mean_squared_error(y, yhat)
    r2 = r2_score(y, yhat)

    res = stats.pearsonr(x_1dim, y_1dim)
    
    if output_file is not None:
        output_file = os.path.join(basepath, output_file)
        with open(output_file, "w") as f:
            f.write(f'Slope: {regression_model.coef_}\n')
            f.write(f'Intercept: {regression_model.intercept_}\n')
            f.write(f'Root mean squared error: {rmse}\n')
            f.write(f'R2 score: {r2}\n')
            f.write(f'{res}\n')
    
    # printing values
    print(f'Slope:' ,regression_model.coef_)
    print(f'Intercept:', regression_model.intercept_)
    print(f'Root mean squared error: ', rmse)
    print(f'R2 score: ', r2)
    print(res)



    # predicted values
    ax.plot(x, yhat, color='r')

def plot_ac_x_cscore(x, y, filename, filenamesuffix, title):
    # x = ac.values
    # y = cscore.values
    xy = np.vstack([x,y])
    z = gaussian_kde(xy, bw_method=1)(xy)
    

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    fig, ax = plt.subplots()
    
    #ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    #ax.scatter_density(x, y, color="red")

    cax = ax.scatter(x, y, c=z, s=50, cmap="viridis")
    fig.colorbar(cax)

    plot_regline(x, y, ax, output_file=f"{filename}_reports_regline_{filenamesuffix}.txt")

    #ax.boxplot([grouped.get_group(g)['cscore'] for g in grouped.groups.keys()])
    #print(f"groups {grouped.groups.keys()}")
    
    #ax.set_xticklabels(grouped.groups.keys())
    ax.set_xlabel("Agreement Complement")
    ax.set_ylabel("c-score")  
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(os.path.join(basepath, f"{filename}{filenamesuffix}.png"))
    plt.close()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--report-disagreement', type=str,
                        help='The report disagreement csv to be processed.')
    parser.add_argument('--cscores-reference', type=str,
        help="The cscores numpy file with cscores for samples in the report"\
        "disagreement csv.")
    parser.add_argument('--dataset-path', type=str,
        help="A folder with each corresponding images from "
        "the dataset corresponding to the input report-disagreement csv.")
    parser.add_argument('--index-file-clean', type=str,
        help="(optional) An index file to convert clean indexes in the disagreement"\
        "csv to match indexes in the cscore numpy file.")
    parser.add_argument('--index-file-unknown', type=str,
        help="(optional) An index file to convert unknown indexes in the disagreement"\
        "csv to match indexes in the cscore numpy file.")
    

    args = parser.parse_args()
    
    dataset_path = args.dataset_path
    rdpath = args.report_disagreement 
    cspath = args.cscores_reference
    idxcpath = args.index_file_clean
    idxupath = args.index_file_unknown

    basepath, subfolder = os.path.split(dataset_path)
    agreed_cscore_one = os.path.join(basepath, f"{subfolder}_agreed_cscore_one")

    os.makedirs(agreed_cscore_one, exist_ok=True)

    rd = pd.read_csv(rdpath)
    rd = rd.set_index('images')
    idxcfile = np.load(idxcpath)
    idxufile = np.load(idxupath)
    idxfile = np.concatenate([idxcfile, idxufile])
    cs = np.load(cspath)
    exp_cols = [i for i in rd.columns.tolist() if i.startswith("exp")]

    majority = []
    for i, row in enumerate(rd[exp_cols].values):
        idxs, counts = np.unique(row, return_counts=True)
        majority.append(idxs[counts.argmax()])
    rd['majority'] = majority

    rd = rd.reindex(idxfile)
    rd['cscore'] = cs
    rd = rd.sort_values(by='cscore')
    basepath, basename = os.path.split(rdpath)
    filename, extension = os.path.splitext(basename)
    rd.to_csv(os.path.join(basepath, f"{filename}_cscore{extension}"))
    
    # dn = dataset_name
    for dn in ["all", "no_unknown", "unknown"]:
        if dn == "all":
            rddn = rd
        else:
            rddn = rd[rd["dataset_name"] == dn]

        ac = rddn['agreement_complement']
        cscore = rddn['cscore']
        label = rddn['label']
        majority = rddn['majority']
        major_wrong = rddn['majority'] != rddn['label']
        major_correct = rddn['majority'] == rddn['label']

        # Copy images from zero ac one cscore to new folders
        if dn == "all":
            for index, row in tqdm(rddn.iterrows(), total=len(rddn)):
                if (row["cscore"] == 1.0) and (row['agreement_complement'] == 0):
                    shutil.copy(os.path.join(dataset_path,f"{index}_{row['label']}.png"), agreed_cscore_one)
        


        # Scatterplot colored by density
        # Ref: https://stackoverflow.com/a/20107592

        #rddn = rddn[rddn['agreement_complement'] > 0]
        #grouped = rddn.groupby('agreement_complement')
        
        #x = ac[ac > 0].values
        #y = cscore[ac > 0].values

        plot_ac_x_cscore(
            x=ac.values,
            y=cscore.values,
            filename=filename,
            filenamesuffix=f"_ac_x_cscore_{dn}",
            title=f"Agreement Complement x c-score - {dn}")
        rddn.to_csv(os.path.join(basepath, f"{filename}_cscore_{dn}{extension}"))
        plot_ac_x_cscore(
            ac[major_wrong].values,
            cscore[major_wrong].values,
            filename,
            f"_ac_wrong_x_cscore_{dn}",
            f"Agreement Complement (Wrong) x c-score - {dn}")
        rddn.to_csv(os.path.join(basepath, f"{filename}_wrong_cscore_{dn}{extension}"))
        plot_ac_x_cscore(
            ac[major_correct].values,
            cscore[major_correct].values,
            filename,
            f"_ac_correct_x_cscore_{dn}",
            f"Agreement Complement (Correct) x c-score - {dn}")
        rddn.to_csv(os.path.join(basepath, f"{filename}_correct_cscore_{dn}{extension}"))
        

        


        
        
