import os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from timeit import default_timer as timer
from utils.neptune_logger import NeptuneLogger
from utils.iteration_criterion import shannon_entropy, EntropyCriterion
from tqdm import tqdm
import matplotlib.pyplot as plt

import logging
from train.focal_loss import FocalLoss

bar_width = 0.5

def train(args, model, device, train_loader, optimizer, epoch, nlogger, use_fsdp=False):
    model.train()
    start = timer()
    epoch_loss = 0
    epoch_entropy = 0
    it_epoch_entropy = {}
    correct = 0
    start_epoch = timer()
    print(f"test nlogger: {nlogger}")
    try:
        if args.loss == 'nll':
            loss_func = F.nll_loss
        elif args.loss == 'focal':
            loss_func = FocalLoss()
        else:
            print(f"args.loss not found: {args.loss}. Using nll_loss.")
            loss_func = F.nll_loss
    except:
        print("args.loss error. Using nll_loss.")
        loss_func = F.nll_loss


    for batch_idx, (data, *target) in enumerate(train_loader):
        if len(target) == 1:
            target = target[0]
            image_names = None
        elif len(target) == 2:
            # This is important when testing the Trees dataset
            image_names = target[1]
            target = target[0]
        if not use_fsdp:
            data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        start = timer()
        output, intermediate_outputs = model(data, gt=target)
        end = timer()
        train_batch_time = end-start
        if nlogger is not None:
            nlogger.log_train_batch_time(end-start)
        output = output.to(target.device)
        loss = loss_func(output, target)
        epoch_loss += loss
        it_batch_entropy = []
        for i, it_out in enumerate(intermediate_outputs):
            it_out = it_out.to(target.device)
            it_loss = loss_func(it_out, target)
            it_entropy = shannon_entropy(it_out)
            loss += it_loss
            it_batch_entropy.append(it_entropy)
            if it_epoch_entropy.get(i) is None:
                it_epoch_entropy[i] = it_entropy*len(it_out)
            else:
                it_epoch_entropy[i] += it_entropy*len(it_out)
        if nlogger is not None:
            nlogger.log_train_batch_it_entropy(it_batch_entropy)
        

        loss = loss/(len(intermediate_outputs)+1)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        batch_correct = pred.eq(target.view_as(pred)).sum().item()
        correct += batch_correct
        loss.backward()
        optimizer.step()
        entropy = shannon_entropy(output)
        epoch_entropy += entropy*len(output)
        if nlogger is not None:
            nlogger.log_train_batch_correct(batch_correct/len(target))
            nlogger.log_train_batch_entropy(entropy)
        if batch_idx % args.log_interval == 0:
            print(
                (f"Train Epoch: {epoch} ",
                f"[{batch_idx * len(data)}/{len(train_loader.dataset)}",
                f" ({len(train_loader.dataset):.0f})]",
                f" Loss: {loss.item():.6f}",
                f" Entropy: {entropy:.3f}",
                f" Time: {(train_batch_time):.3f}")
                )
            logging.info(
                (f"Train Epoch: {epoch} ",
                f"[{batch_idx * len(data)}/{len(train_loader.dataset)}",
                f" ({len(train_loader.dataset):.0f})]",
                f" Loss: {loss.item():.6f}",
                f" Entropy: {entropy:.3f}",
                f" Time: {(end-start):.3f}")
            )
            if args.dry_run:
                break
    end_epoch = timer()
    if nlogger is not None:
        nlogger.log_train_time(end_epoch-start_epoch)
    epoch_entropy /= len(train_loader.dataset)
    it_epoch_entropy = list(it_epoch_entropy.values())
    for i in range(len(it_epoch_entropy)):
        it_epoch_entropy[i] /= len(train_loader.dataset)
    if nlogger is not None:
        nlogger.log_train_entropy(epoch_entropy)
        nlogger.log_train_it_entropy(it_epoch_entropy)
        nlogger.log_train_correct(correct/len(train_loader.dataset))



def log_it_data(log_it_folder, batch, last_exit, output,
                intermediate_outputs, gt, image_names):
    global bar_width
    last_exit = str(last_exit)
    exit_folder = os.path.join(log_it_folder, last_exit)
    os.makedirs(exit_folder, exist_ok=True)
    prev_exits_folder = os.path.join(log_it_folder, last_exit, 'prev_exits')
    os.makedirs(prev_exits_folder, exist_ok=True)
    exit_report_file = os.path.join(log_it_folder, last_exit, 'report.csv')
    if not os.path.exists(exit_report_file):
        with open(exit_report_file, 'w') as f:
            f.write("image name, last exit, entropy, gt, pred, correct\n")

    with open(exit_report_file, 'a') as f:
        for i in range(len(batch)):
            image = batch[i]
            image_prob = torch.nn.functional.softmax(output[i])
            image_gt = gt[i]
            image_pred = torch.argmax(image_prob)
            entropy = shannon_entropy(image_prob.unsqueeze(0), from_logits=False)
            correct = 1 if image_pred == image_gt else 0
            if image_names is not None:
                image_name = str(image_names[i])
            #probabilities plot
            #QuickRef: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html
            x_axis = list(range(1, 1+len(image_prob)))
            y_axis = image_prob.cpu().detach().numpy()
            fig, ax = plt.subplots(figsize=(18,5))
            bar_container = plt.bar(x=x_axis, height=y_axis, color='black', width=bar_width)
            if correct == 1:
                bar_container[image_pred].set_color('green')
            else:
                bar_container[image_pred].set_color('red')
                bar_container[image_gt].set_color('blue')
            plt.savefig(os.path.join(exit_folder, f"{image_name}_probs_plot.png"))
            plt.close()

            f.write(f"{image_name}, {last_exit}, {entropy}, {image_gt}, {image_pred}, {correct}\n")
            save_image(image, os.path.join(exit_folder, f"{os.path.split(image_name)[-1]}.png"))
            for j in range(len(intermediate_outputs)):
                image_it_prob = torch.nn.functional.softmax(intermediate_outputs[j][i], dim=0)
                image_it_pred = torch.argmax(image_it_prob)
                image_gt = gt[i]
                image_pred = torch.argmax(image_it_prob)
                entropy = shannon_entropy(image_it_prob.unsqueeze(0), from_logits=False)
                correct = 1 if image_pred == image_gt else 0
                if image_names is not None:
                    image_name = str(image_names[i])
                #probabilities plot
                #QuickRef: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html
                x_axis = list(range(1, 1+len(image_prob)))
                y_axis = image_it_prob.cpu().detach().numpy()
                fig, ax = plt.subplots(figsize=(18,5))
                bar_container = plt.bar(x=x_axis, height=y_axis, color='black', width=bar_width)
                if correct == 1:
                    bar_container[image_pred].set_color('green')
                else:
                    bar_container[image_pred].set_color('red')
                    bar_container[image_gt].set_color('blue')
                plt.savefig(os.path.join(prev_exits_folder, f"{os.path.split(image_name)[-1]}_probs_plot_it_{j}.png"))
                plt.close()

def test(model, device, test_loader, nlogger=None, log_it_folder=None, use_fsdp=False):
    model.eval()
    test_loss = 0
    test_entropy = 0
    correct = 0
    it_epoch_entropy = {}
    it_epoch_correct = {}
    start_epoch = timer()
    last_image_idx = 0
    print(f"test nlogger: {nlogger}")
    with torch.no_grad():
        for data, *target in tqdm(test_loader):
            if len(target) == 1:
                target = target[0]
                image_names = None
            elif len(target) == 2:
                # This is important when testing the Trees dataset
                image_names = target[1]
                target = target[0]
            else:
                raise Exception("The number of values unpacked from test_loader must be 1 or 2.")

            if not use_fsdp:
                data, target = data.to(device), target.to(device)
            
            #output, intermediate_outputs = model(data, EntropyCriterion(gt=target))
            start = timer()
            output, intermediate_outputs = model(data, test=True)
            end = timer()
            pred = output.argmax(dim=1, keepdim=True)
            if nlogger is not None:
                nlogger.log_test_batch_time(end-start)
            last_exit = model.get_last_exit()
            test_batch_entropy = shannon_entropy(output)
            test_entropy += test_batch_entropy*len(output)
            if nlogger is not None:
                nlogger.log_test_last_exit(last_exit)
                nlogger.log_test_batch_entropy(test_batch_entropy)

            batch_it_entropies = []
            batch_it_corrects = []
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            for i, it_out in enumerate(intermediate_outputs):
                it_loss = F.nll_loss(it_out, target, reduction='sum').item()
                batch_it_entropy = shannon_entropy(it_out)
                test_loss += it_loss
                pred_it = it_out.argmax(dim=1, keepdim=True)
                
                # The mean is taken because the accuracy of the whole batch is considered
                batch_it_correct = pred_it.eq(target.view_as(pred_it)).sum().item()/len(it_out)
                
                batch_it_entropies.append(batch_it_entropy)
                batch_it_corrects.append(batch_it_correct)
                if it_epoch_correct.get(i) is None:
                    it_epoch_correct[i] = batch_it_correct*len(it_out)
                else:
                    it_epoch_correct[i] += batch_it_correct*len(it_out)
                if it_epoch_entropy.get(i) is None:
                    it_epoch_entropy[i] = batch_it_entropy*len(it_out)
                else:
                    it_epoch_entropy[i] += batch_it_entropy*len(it_out)

            test_loss = test_loss/(len(intermediate_outputs)+1)
            batch_correct = pred.eq(target.view_as(pred)).sum().item()
            if nlogger is not None:
                nlogger.log_test_batch_it_entropy(batch_it_entropies)
                nlogger.log_test_batch_it_correct(batch_it_corrects)
                nlogger.log_test_batch_correct(batch_correct/len(target))
            
            correct += batch_correct
            print(
                f"Test batch: Average entropy: {test_batch_entropy:.4f}, "
                f"Accuracy: {batch_correct}/{len(data)}"
                f" ({100. * batch_correct / len(data):.0f}%) ")
            it_entropies_str = ", ".join([f"{i:.4f}" for i in batch_it_entropies])
            print(f"Intermediate batch entropies: {it_entropies_str}")
            if log_it_folder is not None:
                os.makedirs(log_it_folder, exist_ok=True)
                if image_names is None:
                    image_names = list(range(last_image_idx, last_image_idx+len(data)))
                log_it_data(log_it_folder, data, last_exit, output, intermediate_outputs, target, image_names=image_names)
                last_image_idx += len(data)
                

                    
    end_epoch = timer()
    if nlogger is not None:
        nlogger.log_test_time(end_epoch-start_epoch)

    test_entropy /= len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    it_epoch_entropy = list(it_epoch_entropy.values())
    for i in range(len(it_epoch_entropy)):
        it_epoch_entropy[i] /= len(test_loader.dataset)
    if nlogger is not None:
        nlogger.log_test_entropy(test_entropy)
        nlogger.log_test_it_entropy(it_epoch_entropy)
        nlogger.log_test_correct(correct/len(test_loader.dataset))


    print(
        f"\nTest: Average loss: {test_loss:.4f}, "
        f"Average entropy: {test_batch_entropy:.4f}, "
        f"Accuracy: {correct}/{len(test_loader.dataset)}"
        f" ({100. * correct / len(test_loader.dataset):.0f}%)\n")

    logging.info(
        f"\nTest set: Average loss: {test_loss:.4f},"
        f" Accuracy: {correct}/{len(test_loader.dataset)}"
        f" ({100. * correct / len(test_loader.dataset):.0f}%)\n")
    return correct