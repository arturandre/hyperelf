import neptune
import datetime


class PrintLogger():
    # self\.run\["?f?"?([\w|/]+)"\].log\((\w+)\)
    # self.log(f"{$1} {$2}")
    def __init__(self, name=None):
        self.logfile = name

    def log(self, msg):
        ctime = datetime.datetime.now()
        with open(self.logfile, "a+") as f:
            f.write(f"{ctime} {msg}\n")

    def setup_neptune(self, nparams = None):
        self.log(f"parameters {nparams}")
        
    def log_train_last_exit(self, last_exit):
        self.log(f"train/batch/last_exit {last_exit}")

    def log_train_batch_time(self, time):
        self.log(f"train/batch/time {time}")

    def log_lr(self, current_lr):
        self.log(f"train/learning_rate {current_lr}")

    def log_train_time(self, time):
        self.log(f"train/time {time}")

    def log_train_entropy(self, entropy):
        self.log(f"train/entropy {entropy}")

    def log_train_batch_entropy(self, entropy):
        self.log(f"train/batch/entropy {entropy}")
        
    def log_train_it_entropy(self, it_entropy):
        for i, entropy in enumerate(it_entropy):
            self.log(f"train/it_entropy_{i} {entropy}")

    def log_train_batch_it_entropy(self, it_entropy):
        for i, entropy in enumerate(it_entropy):
            self.log(f"train/batch/it_entropy_{i} {entropy}")

    def log_train_batch_correct(self, correct):
        self.log(f"train/batch/correct {correct}")

    def log_train_correct(self, correct):
        self.log(f"train/correct {correct}")

    def log_train_batch_it_correct(self, it_correct):
        for i, correct in enumerate(it_correct):
            self.log(f"train/batch/it_correct_{i} {correct}")

    def log_test_last_exit(self, last_exit):
        self.log(f"test/batch/last_exit {last_exit}")

    def log_test_batch_time(self, time):
        self.log(f"test/batch/time {time}")

    def log_test_time(self, time):
        self.log(f"test/time {time}")

    def log_test_batch_entropy(self, entropy):
        self.log(f"test/batch/entropy {entropy}")

    def log_test_batch_it_entropy(self, it_entropy):
        for i, entropy in enumerate(it_entropy):
            self.log(f"test/batch/it_entropy_{i} {entropy}")

    def log_test_entropy(self, entropy):
        self.log(f"test/entropy {entropy}")

    def log_test_it_entropy(self, it_entropy):
        for i, entropy in enumerate(it_entropy):
            self.log(f"test/it_entropy_{i} {entropy}")

    def log_test_batch_correct(self, correct):
        self.log(f"test/batch/correct {correct}")

    def log_test_correct(self, correct):
        self.log(f"test/correct {correct}")

    def log_test_batch_it_correct(self, it_correct):
        for i, correct in enumerate(it_correct):
            self.log(f"test/batch/it_correct_{i} {correct}")

    def stop(self):
        pass


class NeptuneLogger():
    def __init__(self, name=None, tags=None, nparams=None, project=None, api_token=None):
        if project is None:
            project="inacity/HyperElyx"
        if api_token is None:
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2NmZmOTAzYi02NTJkLTQ0MzUtOTYzYi1kYjVjZTVjYzc4MmMifQ=="
        self.run = run = neptune.init_run(
            project=project,
            api_token=api_token,
            name=name,
            tags=tags,
        )  # your credentials


        if nparams is not None:
            self.setup_neptune(nparams)

    def setup_neptune(self, nparams = None):
        self.run["parameters"] = nparams


    def log_train_last_exit(self, last_exit):
        self.run["train/batch/last_exit"].log(last_exit)

    def log_train_batch_time(self, time):
        self.run["train/batch/time"].log(time)

    def log_lr(self, current_lr):
        self.run["train/learning_rate"].log(current_lr)

    def log_train_time(self, time):
        self.run["train/time"].log(time)

    def log_train_entropy(self, entropy):
        self.run["train/entropy"].log(entropy)

    def log_train_batch_entropy(self, entropy):
        self.run["train/batch/entropy"].log(entropy)
        
    def log_train_it_entropy(self, it_entropy):
        for i, entropy in enumerate(it_entropy):
            self.run[f"train/it_entropy_{i}"].log(entropy)

    def log_train_batch_it_entropy(self, it_entropy):
        for i, entropy in enumerate(it_entropy):
            self.run[f"train/batch/it_entropy_{i}"].log(entropy)

    def log_train_batch_correct(self, correct):
        self.run["train/batch/correct"].log(correct)

    def log_train_correct(self, correct):
        self.run["train/correct"].log(correct)

    def log_train_batch_it_correct(self, it_correct):
        for i, correct in enumerate(it_correct):
            self.run[f"train/batch/it_correct_{i}"].log(correct)

    def log_test_last_exit(self, last_exit):
        self.run["test/batch/last_exit"].log(last_exit)

    def log_test_batch_time(self, time):
        self.run["test/batch/time"].log(time)

    def log_test_time(self, time):
        self.run["test/time"].log(time)

    def log_test_batch_entropy(self, entropy):
        self.run["test/batch/entropy"].log(entropy)

    def log_test_batch_it_entropy(self, it_entropy):
        for i, entropy in enumerate(it_entropy):
            self.run[f"test/batch/it_entropy_{i}"].log(entropy)

    def log_test_entropy(self, entropy):
        self.run["test/entropy"].log(entropy)

    def log_test_it_entropy(self, it_entropy):
        for i, entropy in enumerate(it_entropy):
            self.run[f"test/it_entropy_{i}"].log(entropy)

    def log_test_batch_correct(self, correct):
        self.run["test/batch/correct"].log(correct)

    def log_test_correct(self, correct):
        self.run["test/correct"].log(correct)

    def log_test_batch_it_correct(self, it_correct):
        for i, correct in enumerate(it_correct):
            self.run[f"test/batch/it_correct_{i}"].log(correct)



    def stop(self):
        self.run.stop()

# for epoch in range(10):
#     run["train/loss"].log(0.9 ** epoch)

# run["eval/f1_score"] = 0.66

