import argparse
import os

from datasets import load_dataset
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
import torch

from finetune import *

parser = argparse.ArgumentParser(description="Finetuing a Resnet50 model")
parser.add_argument('--eps', type=float, help='maximum perturbation for PGD attack', default=8 / 255)                           # Not Used
parser.add_argument('--alpha', type=float, help='learning rate', default=1e-4)                                                  # Changed from 2 / 255
parser.add_argument('--steps', type=int, help='number of epochs', default=20)                                                   # Used for finetuning
parser.add_argument('--batch_size', type=int, help='batch size for training', default=100)                                      # Used for finetuning
parser.add_argument('--batch_num', type=int, help='number of batches on which to run PGD attack', default=None)                 # Not Used
parser.add_argument('--results', type=str, help='name of the file to save the results to', required=True)                       # Used for saving finetuning results & model weights
parser.add_argument('--resultsdir', type=str, help='name of the folder to save the results to', default='finetune_results')     # Used for finetuning results & model weights 
parser.add_argument('--seed', type=int, help='set manual seed value for reproducibility, default 1234',
                    default=1234)
parser.add_argument('--test', action='store_true', help='test that code runs')                                                  # Not Used
args = parser.parse_args()

RESULTS_DIR = args.resultsdir
RESULTS_PATH = os.path.join(RESULTS_DIR, args.results)
if not os.path.isdir(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

if args.seed:
    SEED = args.seed
    torch.manual_seed(SEED)
else:
    SEED = torch.seed()

EPS = args.eps
ALPHA = args.alpha
STEPS = args.steps

BATCH_SIZE = args.batch_size
BATCH_NUM = args.batch_num
if BATCH_NUM is None:
    BATCH_NUM = 1281167 // BATCH_SIZE + 1
assert BATCH_NUM > 0

print('Loading model...')
# Step 1: Initialize model with the best available weights
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
preprocess = weights.transforms()

# Step 2: Load and preprocess data
print('Loading data...')

# # Load clean ImageNet-1k dataset from Huggingface
ds = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True, trust_remote_code=True)
def preprocess_img(example):
    example['image'] = preprocess(example['image'])
    return example
# Filter out grayscale images
ds = ds.filter(lambda example: example['image'].mode == 'RGB')
# Preprocess function will be applied to images on-the-fly whenever they are being accessed in the loop
ds = ds.map(preprocess_img)
ds = ds.shuffle(seed=SEED)
# Only take desired portion of dataset
ds = ds.take(BATCH_NUM * BATCH_SIZE)
dset_loader = DataLoader(ds, batch_size=BATCH_SIZE)

# Load adversial dataset f"results/gen_data" as trainset (eps=0.1) & f"results/gen_data_003" as testset (eps=8/255)
# ds_adv = AdversialDataset(load=False, path="adv_data" ,eps="gen_data")
# ds_adv_trainset, ds_adv_testset =  train_val_dataset(ds_adv)
ds_adv_trainset, ds_adv_testset = AdversialDataset(load=False, path="results" ,eps="gen_data"),  AdversialDataset(load=False, path="results" ,eps="gen_data_003")
dset_loader_adv_train = DataLoader(ds_adv_trainset, batch_size=64, shuffle=True)
dset_loader_adv_test = DataLoader(ds_adv_testset, batch_size=64, shuffle=True)

# Step 3: Initialize Trainer
trainer = ResnetFinetune(model, dset_loader_adv_test)
trainer.init_metrics()
trainer.use_adv_valid = True
trainer.add_trainloader(dset_loader_adv_train)
trainer.add_optimizer(alpha=ALPHA)

# Step 4: Finetune model & evaluate
print(f"===Full Finetuning on Adverserial Images===")
trainer.finetune_train(EPS, ALPHA, STEPS, BATCH_NUM)

# Step 5: Save finetuning metrics & model weights for post-analysis for overfitting (manual Early stopping & check for Convergence)
torch.save({
    'train_acc': trainer.train_accs,
    'train_loss': trainer.train_losses,
    'test_accs': trainer.test_accs,
    'test_losses':trainer.test_losses,
    'model_weights': trainer.model.state_dict()
}, RESULTS_PATH)