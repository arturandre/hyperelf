import neptune.new as neptune

class NeptuneLogger():
    def __init__(self, name=None, tags=None, nparams=None):
        self.run = run = neptune.init(
            project="inacity/HyperElyx",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2NmZmOTAzYi02NTJkLTQ0MzUtOTYzYi1kYjVjZTVjYzc4MmMifQ==",
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

