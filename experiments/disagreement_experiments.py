# There are multiple ways to compose
# the voting comitee (trained networks whose disagreements will be analysed).

# 1 - For each architecture
#   - Use a single labeling strategy
#   - All with/without early exits
# 2 - Train from scratch the same architecture multiple times,
# 3 - A combination of the previous approaches.

from torchvision.utils import save_image
from argparse import ArgumentParser
import os
import neptune.new as neptune
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm 

import os, sys
sys.path.append( # "."
    os.path.dirname( #"experiments/"
    os.path.dirname( #"hyperelf/" 
        os.path.abspath(__file__))))

from utils.dataset import prepare_dataset

import matplotlib
matplotlib.use('Agg')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('expfolder', default="",
        help="Base path containing the experiment folders")
    parser.add_argument('expnames', default=[], 
        help="Comma separated sequence of experiment folder names inside the experiment base path. It is expected exactly one .pt file in each folder.")
    parser.add_argument('--output-path', type=str,
                        help='(optional) Defines where to create the reports.')
    parser.add_argument('--dataset-name-clean', type=str,
        help='The clean dataset')
    parser.add_argument('--dataset-name-unknown', type=str,
        help='The unknown dataset')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--save-disagreements', action='store_true', default=False,
                        help='Save images with disagreements.')
    parser.add_argument('--save-dataset', action='store_true', default=False,
                        help='Save all the dataset images with the png format.')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()
    
    dataset_name_clean = args.dataset_name_clean
    dataset_name_unknown = args.dataset_name_unknown
    basepath = args.expfolder
    expnames = args.expnames.split(",")

    if args.output_path is not None:
        outpath = args.output_path 
        os.makedirs(outpath, exist_ok=True)
    else:
        outpath = ""

    if args.save_dataset:
        imgdatasetpath = os.path.join(outpath, "dataset")
        os.makedirs(imgdatasetpath, exist_ok=True)
    if args.save_disagreements:
        imgdpath = os.path.join(outpath, "imgdisagreements")
        os.makedirs(imgdpath, exist_ok=True)

    

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    #train_kwargs = {'batch_size': args.batch_size}
    #test_kwargs = {'batch_size': args.test_batch_size}
    train_kwargs = {'batch_size': 80, 'shuffle': False}
    test_kwargs = {'batch_size': 80, 'shuffle': False}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)



    # Load each network from .pt files
    # Predict outputs for a given dataset
    # Store the predicted labels
    ## In a dataframe containing the ImageName as index,
    ## and each column has predictions for a given network
    train_loader_clean, test_loader_clean =\
        prepare_dataset(
        dataset_name = dataset_name_clean, # from summanTrain
        use_imagenet_stat = True,
        train_kwargs=train_kwargs,
        test_kwargs=test_kwargs,
    )
    train_loader_unknown, test_loader_unknown =\
        prepare_dataset(
        dataset_name = dataset_name_unknown, # from summanTrain too (but with labelmode="unknown")
        use_imagenet_stat = True,
        train_kwargs=train_kwargs,
        test_kwargs=test_kwargs,
    )
    combined_df = {
        "train": None,
        "test": None        
    }
    #
    # Do something similar to combined_df and exp_df
    # for the predictions on the training datasets
    # so that experiments can be done with images
    # predicted to be unknown/clean.

    combined_loaded = False
    first_image = None
    second_image = None
    first_label = None
    second_label = None
    test_datasets = [
        ("no_unknown", test_loader_clean),
        ("unknown", test_loader_unknown)
    ]
    train_datasets = [
            ("no_unknown", train_loader_clean),
            ("unknown", train_loader_unknown)
        ]
    for expname in expnames:
        index_start = 0
        fullpath = os.path.join(basepath, expname)
        files = os.listdir(fullpath)
        pt_files = [f for f in files if f.endswith(".pt")]
        if len(pt_files) != 1:
            raise RuntimeError(
                f"{len(pt_files)} .pt files found at: {fullpath}. "
                f"Exactly one should be present at each experiment folder."
                )
        model = torch.load(os.path.join(fullpath, pt_files[0]))
        model.eval()
        with torch.no_grad():
            exp_df = {
                "train": None,
                "test": None
                }
            for stage_name, dataset_stage in zip(
                ["train", "test"],
                [train_datasets, test_datasets]
            ):
                if os.path.exists(os.path.join(outpath, f'report_disagreements_{stage_name}.csv')):
                    combined_df[stage_name] = pd.read_csv(os.path.join(outpath, f'report_disagreements_{stage_name}.csv'))
                    combined_df[stage_name] = combined_df[stage_name].set_index('images')
                    combined_loaded = True
                    continue
                for dataset_name, datasets in dataset_stage:
                    all_names = []
                    all_preds = None
                    all_labels = None
                    for data, *target in tqdm(datasets):
                        if len(target) == 1:
                            target = target[0]
                            image_names = list(range(index_start, index_start+len(target)))
                            index_start += len(target)
                        elif len(target) == 2:
                            # This is important when testing the Trees dataset
                            image_names = target[1]
                            target = target[0]
                        else:
                            raise Exception(
                                f"The number of values unpacked" 
                                f"from test_loader must be 1 or 2."
                                )
                        #data, target = data.to(device), target.to(device)
                        data = data.to(device)
                        output, intermediate_outputs = model(data, test=True)
                        pred = output.argmax(dim=1, keepdim=True)
                        all_names += image_names
                        if all_preds is None:
                            all_preds = pred
                        else:
                            all_preds = torch.concat([all_preds, pred])
                        if all_labels is None:
                            all_labels = target
                        else:
                            all_labels = torch.concat([all_labels, target])
                    all_preds = all_preds.detach().cpu().numpy().reshape(-1)
                    aux = pd.DataFrame({"images": all_names})
                    aux = aux.set_index("images")
                    aux.loc[:, "dataset_name"] = dataset_name
                    aux.loc[:, "label"] = all_labels.detach().cpu().numpy().reshape(-1)
                    aux.loc[:, expname] = all_preds
                    if exp_df[stage_name] is None:
                        exp_df[stage_name] = aux
                    else:
                        exp_df[stage_name] = pd.concat([exp_df[stage_name], aux])
                if combined_df[stage_name] is None:
                    combined_df[stage_name] = exp_df[stage_name]
                else:
                    combined_df[stage_name] = pd.concat([combined_df[stage_name], exp_df[stage_name].loc[:, expname]],
                        axis=1, join='inner')
    for stage_name in ["train", "test"]:
    #for stage_name in ["test"]:
        #if combined_loaded:
        #    break
        #combined_df[stage_name].loc[:, "disagreement"] = combined_df[stage_name].loc[:, expnames].sum(axis=1)/len(expnames)
        # AC - # Agreements complement (total - max agreements)
        # DL - # Disagreement Levels (max agreement/number of decisions)
        ACs = []
        DLs = []
        for row in combined_df[stage_name].loc[:, expnames].values:
            unique, counts = np.unique(row, return_counts=True)
            ac = len(expnames) - np.max(counts)
            dl = 1 - np.max(counts)/len(expnames)
            ACs.append(ac)
            DLs.append(dl)
        combined_df[stage_name].loc[:, "disagreement"] = DLs
        combined_df[stage_name].loc[:, "agreement_complement"] = ACs
        if not combined_loaded:
            combined_df[stage_name].to_csv(os.path.join(outpath, f'report_disagreements_{stage_name}.csv'))
        if args.save_dataset or args.save_disagreements:
            # Here we reload the datasets but
            # without ImageNet statistics
            train_loader_clean, test_loader_clean =\
                prepare_dataset(
                dataset_name = dataset_name_clean, # from summanTrain
                use_imagenet_stat = False,
                train_kwargs=train_kwargs,
                test_kwargs=test_kwargs,
            )
            train_loader_unknown, test_loader_unknown =\
                prepare_dataset(
                dataset_name = dataset_name_unknown, # from summanTrain too (but with labelmode="unknown")
                use_imagenet_stat = False,
                train_kwargs=train_kwargs,
                test_kwargs=test_kwargs,
            )
        if args.save_dataset:
            imgdatasetpath_ = os.path.join(imgdatasetpath, stage_name)
            os.makedirs(imgdatasetpath_, exist_ok=True)
            scanned_dataset_loader = iter(train_loader_clean) if stage_name == "train" else iter(test_loader_clean)
            changed_to_unknown_loader = False
            previous_batch_index = 0
            first_index = combined_df[stage_name].iloc[0].name
            last_index = 0
            current_batch = next(scanned_dataset_loader)
            for index, row in tqdm(combined_df[stage_name].iterrows(), total=len(combined_df[stage_name])):
                batch_index = (index-first_index-last_index) // train_kwargs['batch_size']
                img_index = (index-first_index-last_index) % train_kwargs['batch_size']
                # Check if len(scanned_dataset_loader) == len(train_loader_clean) 
                if (batch_index >= len(scanned_dataset_loader))\
                    or ( #Last batch can be smaller
                        (img_index >= len(current_batch[0])
                         or last_index > 0)
                        ):
                    if (img_index == len(current_batch[0])
                         or last_index > 0): #Last batch is equal all other batches
                        batch_index -= (len(scanned_dataset_loader)-1)
                    else: # Last batch is smaller
                        batch_index -= len(scanned_dataset_loader)
                    if not changed_to_unknown_loader:
                        last_index = len(current_batch[0])
                        img_index = 0
                        scanned_dataset_loader = iter(train_loader_unknown)  if stage_name == "train" else iter(test_loader_unknown)
                        current_batch = next(scanned_dataset_loader)
                        previous_batch_index = 0
                        changed_to_unknown_loader = True
                for i in range(previous_batch_index, batch_index):
                    current_batch = next(scanned_dataset_loader)
                images, labels = current_batch
                current_image = images[img_index]
                current_label = labels[img_index]
                save_path = os.path.join(imgdatasetpath_, f"{index-first_index}_{str(current_label.numpy())}.png")
                if not os.path.exists(save_path):
                    save_image(current_image, save_path)
                
                previous_batch_index = batch_index
        if args.save_disagreements:
            imgdpath_ = os.path.join(imgdpath, stage_name)
            os.makedirs(imgdpath_, exist_ok=True)
            disimages = combined_df[stage_name][np.array(ACs) > 0]
            scanned_dataset_loader = iter(train_loader_clean) if stage_name == "train" else iter(test_loader_clean)
            changed_to_unknown_loader = False
            previous_batch_index = 0
            last_index = 0
            first_index = combined_df[stage_name].iloc[0].name
            current_batch = next(scanned_dataset_loader)
            for index, row in tqdm(disimages.iterrows(), total=len(disimages)):
                batch_index = (index-first_index-last_index) // train_kwargs['batch_size']
                img_index = (index-first_index-last_index) % train_kwargs['batch_size']
                # Important for smaller last batches
                #if batch_index >= len(scanned_dataset_loader)-1:
                if not changed_to_unknown_loader:
                    for i in range(previous_batch_index, batch_index-1):
                        current_batch = next(scanned_dataset_loader)
                if (batch_index >= len(scanned_dataset_loader))\
                    or ( #Last batch can be smaller
                        (img_index >= len(current_batch[0])
                         or last_index > 0)
                        ):
                    if (img_index >= len(current_batch[0])
                         or last_index > 0): #Last batch is equal all other batches
                        batch_index -= (len(scanned_dataset_loader)-1)
                    else: # Last batch is smaller
                        batch_index -= len(scanned_dataset_loader)
                    if not changed_to_unknown_loader:
                        last_index = len(current_batch[0])
                        img_index = (index-first_index-last_index) % train_kwargs['batch_size']
                        scanned_dataset_loader = iter(train_loader_unknown)  if stage_name == "train" else iter(test_loader_unknown)
                        current_batch = next(scanned_dataset_loader)
                        previous_batch_index = 0
                        changed_to_unknown_loader = True
                #if batch_index < len(scanned_dataset_loader)-1:
                if changed_to_unknown_loader:
                    for i in range(previous_batch_index, batch_index):
                        current_batch = next(scanned_dataset_loader)
                #if batch_index >= len(scanned_dataset_loader):
                #    for i in range(previous_batch_index, batch_index):
                #        current_batch = next(scanned_dataset_loader)
                images, labels = current_batch
                current_image = images[img_index]
                current_label = labels[img_index]
                save_path = os.path.join(imgdpath_, f"{index-first_index}_{str(current_label.numpy())}.png")
                if not os.path.exists(save_path):
                    save_image(current_image, save_path)
                
                previous_batch_index = batch_index

        unanimous = (combined_df[stage_name].loc[:, "disagreement"]%1).values == 0
        clean = (combined_df[stage_name].loc[:, "dataset_name"] == "no_unknown").values
        clean_d = combined_df[stage_name][clean].loc[:, "disagreement"]
        nbins = int(np.ceil((len(expnames)+1)/2))
        hist, edges = np.histogram(clean_d, bins=nbins)
        bin_centers = ([i/(nbins) for i in range(nbins)])
        bars = plt.bar(bin_centers, 100*hist/hist.sum(), width=1/(nbins*2))
        plt.bar_label(bars, labels=[f"{i:.2f}%" for i in 100*hist/hist.sum()])
        plt.xticks(bin_centers, [f"{i}/{len(expnames)}" \
            for i in range((nbins))], rotation='vertical')
        #plt.xlabel("Positive / Negative votes")
        plt.xlabel("Disagreement Level")
        plt.ylabel("Percentage of images (%)")
        plt.title("Disagreement Levels for clean images")
        plt.tight_layout()
        plt.savefig(os.path.join(outpath, f"report_disagreements_clean_hist_{stage_name}.png"))
        plt.close()

        unknown_d = combined_df[stage_name][~clean].loc[:, "disagreement"]
        hist, edges = np.histogram(unknown_d, bins=nbins)
        bin_centers = ([i/(nbins) for i in range(nbins)])
        bars = plt.bar(bin_centers, 100*hist/hist.sum(), width=1/(nbins*2))
        plt.bar_label(bars, labels=[f"{i:.2f}%" for i in 100*hist/hist.sum()])
        plt.xticks(bin_centers, [f"{i}/{len(expnames)}" \
            for i in range((nbins))], rotation='vertical')
        plt.xlabel("Disagreement Level")
        plt.ylabel("Percentage of images (%)")  
        plt.title("Disagreement Levels for unknown images")
        plt.tight_layout()
        plt.savefig(os.path.join(outpath, f"report_disagreements_unknown_hist_{stage_name}.png"))
        plt.close()

        with open(os.path.join(outpath, f"report_disagreements_{stage_name}.txt"), "w") as f:
            f.write(f"Input arguments: {args}\n\n")

            f.write(f"Total clean: {len(clean)}\n")
            f.write(f"Num. disagreements clean: {len(clean_d) - len(combined_df[stage_name][clean & unanimous])}\n")
            f.write(f"Mean disagreements clean: {clean.mean()}\n")
            f.write(f"Std. disagreements clean: {clean.std()}\n\n")
            f.write(f"Total unknown: {len(unknown_d)}\n")
            f.write(f"Num. disagreements unknown: {len(unknown_d) - len(combined_df[stage_name][~clean & unanimous])}\n")
            f.write(f"Mean disagreements unknown: {unknown_d.mean()}\n")
            f.write(f"Std. disagreements unknown: {unknown_d.std()}\n")



            

        

