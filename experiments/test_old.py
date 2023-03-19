def test(model, device, test_loader, nlogger, log_it_folder=None):

    def log_it_data(log_it_folder, batch, last_exit, probs, gt, image_names):
        last_exit = str(last_exit)
        exit_folder = os.path.join(log_it_folder, last_exit)
        os.makedirs(exit_folder, exist_ok=True)
        exit_report_file = os.path.join(log_it_folder, last_exit, 'report.csv')
        if not os.path.exists(exit_report_file):
            with open(exit_report_file, 'w') as f:
                f.write("image name, last exit, entropy, gt, pred, correct\n")
        with open(exit_report_file, 'a') as f:
            for i in range(len(batch)):
                image = batch[i]
                image_prob = probs[i]
                image_gt = gt[i]
                image_pred = torch.argmax(image_prob)
                entropy = shannon_entropy(image_prob.unsqueeze(0))
                correct = 1 if image_pred == image_gt else 0
                if image_names is not None:
                    image_name = image_names[i]
                
                #probabilities plot
                #QuickRef: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html
                x_axis = list(range(len(image_prob)))
                y_axis = image_prob
                bar_container = plt.bar(x=x_axis, heigh=y_axis)
                if correct == 1:
                    bar_container[image_pred].color = 'green'
                else:
                    bar_container[image_pred].color = 'red'
                    bar_container[image_gt].color = 'blue'
                plt.savefig(os.path.join(exit_folder, f"{image_name}_probs_plot.png"))

                f.write(f"{image_name}, {last_exit}, {entropy}, {image_gt}, {image_pred}, {correct}\n")
                save_image(image, os.path.join(exit_folder, f"{image_name}.png"))

    model.eval()
    test_loss = 0
    test_entropy = 0
    correct = 0
    it_epoch_entropy = {}
    it_epoch_correct = {}
    start_epoch = timer()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            #output, intermediate_outputs = model(data, EntropyCriterion(gt=target))
            start = timer()
            output, intermediate_outputs = model(data, test=True)
            end = timer()
            pred = output.argmax(dim=1, keepdim=True)
            nlogger.log_test_batch_time(end-start)
            last_exit = model.get_last_exit()
            nlogger.log_test_last_exit(last_exit)
            test_batch_entropy = shannon_entropy(output)
            nlogger.log_test_batch_entropy(test_batch_entropy)
            test_entropy += test_batch_entropy*len(output)

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
                
                batch_it_entropies.append(f"{batch_it_entropy:.4f}")
                batch_it_corrects.append(f"{batch_it_correct:.4f}")
                if it_epoch_correct.get(i) is None:
                    it_epoch_correct[i] = batch_it_correct*len(it_out)
                else:
                    it_epoch_correct[i] += batch_it_correct*len(it_out)
                if it_epoch_entropy.get(i) is None:
                    it_epoch_entropy[i] = batch_it_entropy*len(it_out)
                else:
                    it_epoch_entropy[i] += batch_it_entropy*len(it_out)
            nlogger.log_test_batch_it_entropy(batch_it_entropies)
            nlogger.log_test_batch_it_correct(batch_it_corrects)

            test_loss = test_loss/(len(intermediate_outputs)+1)
            batch_correct = pred.eq(target.view_as(pred)).sum().item()
            nlogger.log_test_batch_correct(batch_correct/len(target))
            
            correct += batch_correct
            print(
                f"Test batch: Average entropy: {test_batch_entropy:.4f}, "
                f"Accuracy: {batch_correct}/{len(data)}"
                f" ({100. * batch_correct / len(data):.0f}%)")
            it_entropies_str = " , ".join(batch_it_entropies)
            print(f"Intermediate batch entropies: {it_entropies_str}")
            if log_it_folder is not None:
                os.makedirs(log_it_folder, exist_ok=True)
                log_it_data(log_it_folder, data, last_exit, output, target)