import json
from gliner.model import GLiNER

import torch
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from datasets import load_dataset
import os
from types import SimpleNamespace

os.environ['TASK'] = 'ner'


train_path = "sample_data.json" # from https://github.com/urchade/GLiNER/blob/main/examples/sample_data.json

with open(train_path, "r") as f:
    data = json.load(f)


model = GLiNER.from_pretrained("urchade/gliner_small")


# Define the hyperparameters in a config variable
config = SimpleNamespace(
    num_steps=1000, # number of training iteration
    train_batch_size=2, 
    eval_every=100, # evaluation/saving steps
    save_directory="logs", # where to save checkpoints
    warmup_ratio=0.1, # warmup steps
    device='cpu',
    lr_encoder=1e-5, # learning rate for the backbone
    lr_others=5e-5, # learning rate for other parameters
    freeze_token_rep=False, # freeze of not the backbone
    
    # Parameters for set_sampling_params
    max_types=25, # maximum number of entity types during training
    shuffle_types=True, # if shuffle or not entity types
    random_drop=True, # randomly drop entity types
    max_neg_type_ratio=1, # ratio of positive/negative types, 1 mean 50%/50%, 2 mean 33%/66%, 3 mean 25%/75% ...
    max_len=384 # maximum sentence length
)


# Set sampling parameters from config
model.set_sampling_params(
    max_types=config.max_types, 
    shuffle_types=config.shuffle_types, 
    random_drop=config.random_drop, 
    max_neg_type_ratio=config.max_neg_type_ratio, 
    max_len=config.max_len
)

train_loader = model.create_dataloader(data, batch_size=5, shuffle=False)
iter_train_loader = iter(train_loader)
x = next(iter_train_loader)

# import ipdb;ipdb.set_trace()

loss = model(x)
loss