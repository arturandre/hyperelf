import os
from utils.dataset import get_dataset_info, prepare_dataset, get_dataset_stats
from argparse import Namespace
args = {
  'tophalf': False,
  'tophalfresize': False,
  'tophalfresizehorizontal': False
  }
args = Namespace(**args)

train_kwargs = {'pin_memory': True,
                'batch_size': 2048, 'shuffle': True}
test_kwargs = {'pin_memory': True,
               'batch_size': 2048, 'shuffle': True}

pseudo_loader, _ = prepare_dataset(
  dataset_name="TreesUnlabed40k",
  ignore_labels=True,
  use_imagenet_stat=False,
  train_kwargs=train_kwargs, 
  test_kwargs=test_kwargs,
  fullres=True,
  args=args
)

train_loader, test_loader =\
	prepare_dataset(
	dataset_name = "TreesLocatingTest3k",
    ignore_labels=True,
	extra_train_loader=[pseudo_loader],
	use_imagenet_stat = False,
	train_kwargs=train_kwargs,
	test_kwargs=test_kwargs,
	fullres=True,
	args=args,
)

mean, std = get_dataset_stats(train_loader)
with open('trees_stats.txt', 'w') as f:
    f.write(f"mean: {mean}\n")
    f.write(f"std: {std}\n")