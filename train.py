import argparse
import os

import torch
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

# from model_nested import NerFilteredSemiCRF
from gliner import GLiNER
from gliner.modules.run_evaluation import sample_train_data
from gliner.model import load_config_as_namespace
from datetime import datetime
import json
import logging


logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

'''

python train.py --config config_small_rel.yaml --log_dir logs --relation_extraction

'''

# train function
def train(model, optimizer, train_data, eval_data=None, num_steps=1000, eval_every=100, log_dir=None, wandb_log=False, warmup_ratio=0.1,
          train_batch_size=8, device='cuda'):
    
    if wandb_log:
        import wandb
        # Start a W&B Run with wandb.init
        wandb.login()
        run = wandb.init(project="GLiREL")
    else:
        run = None
    
    if log_dir is None:
        current_time = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        log_dir = f'logs/log-{current_time}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # set up logging
    log_file = "train.log"
    log_file_path = os.path.join(log_dir, log_file)
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    model.train()

    # initialize data loaders
    train_loader = model.create_dataloader(train_data, batch_size=train_batch_size, shuffle=False)

    pbar = tqdm(range(num_steps))

    if warmup_ratio < 1:
        num_warmup_steps = int(num_steps * warmup_ratio)
    else:
        num_warmup_steps = int(warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_steps
    )

    iter_train_loader = iter(train_loader)

    for step in pbar:
        try:
            x = next(iter_train_loader)
        except StopIteration:
            iter_train_loader = iter(train_loader)
            x = next(iter_train_loader)

        for k, v in x.items():
            if isinstance(v, torch.Tensor):
                x[k] = v.to(device)

        try:
            loss = model(x)  # Forward pass
        except Exception as e:
            logger.error(f"Error in step {step}: {e}")
            continue

        logger.info(f"Step {step} | x['rel_label']: {x['rel_label'].shape} | x['tokens']: {len(x['tokens'])} | x['span_idx']: {x['span_idx'].shape} | loss: {loss.item()} | candidate_classes: {x['classes_to_id']}")
        

        # check if loss is nan
        if torch.isnan(loss):
            continue

        loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters
        scheduler.step()  # Update learning rate schedule
        optimizer.zero_grad()  # Reset gradients

        description = f"step: {step} | epoch: {step // len(train_loader)} | loss: {loss.item():.2f}"

        if run is not None:
            run.log({"loss": loss.item()})

        if (step + 1) % eval_every == 0:

            logger.info('Evaluating...')

            model.eval()
            
            if eval_data is not None:
                results, f1 = model.evaluate(
                    eval_data, 
                    flat_ner=True, 
                    threshold=0.5, 
                    batch_size=12,
                    # NOTE: we use collate_fn's ability to negative sample
                    # instead of giving labels ourselves
                    #  entity_types=eval_data["entity_types"]
                )

                logger.info(f"Step={step}\n{results}")
            current_path = os.path.join(log_dir, f'model_{step + 1}')
            model.save_pretrained(current_path)
            #val_data_dir =  "/gpfswork/rech/ohy/upa43yu/NER_datasets" # can be obtained from "https://drive.google.com/file/d/1T-5IbocGka35I7X3CE6yKe5N_Xg2lVKT/view"
            #get_for_all_path(model, step, log_dir, val_data_dir)  # you can remove this comment if you want to evaluate the model

            model.train()

        pbar.set_description(description)


def create_parser():
    parser = argparse.ArgumentParser(description="Span-based NER")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument('--log_dir', type=str, default=None, help='Path to the log directory')
    parser.add_argument("--relation_extraction", action="store_true", help="Activate relation extraction mode")
    parser.add_argument("--wandb_log", action="store_true", help="Activate wandb logging")
    return parser


if __name__ == "__main__":
    # parse args
    parser = create_parser()
    args = parser.parse_args()

    if args.relation_extraction is True:
        os.environ['TASK'] = 'rel'
        logger.info("🚀 Relation extraction mode activated")
    else:
        os.environ['TASK'] = 'ner'

    # load config
    config = load_config_as_namespace(args.config)

    config.log_dir = args.log_dir

    try:
        if config.train_data.endswith('.jsonl'):
            with open(config.train_data, 'r') as f:
                data = [json.loads(line) for line in f]
        elif config.train_data.endswith('.json'):
            with open(config.train_data, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Invalid data format: {config.train_data}")
    except:
        data = sample_train_data(config.train_data, 10000)

    if hasattr(config, 'eval_data'):
        try:
            if config.eval_data.endswith('.jsonl'):
                with open(config.eval_data, 'r') as f:
                    eval_data = [json.loads(line) for line in f]
            elif config.eval_data.endswith('.json'):
                with open(config.eval_data, 'r') as f:
                    eval_data = json.load(f)
            else:
                raise ValueError(f"Invalid data format: {config.eval_data}")
        except:
            eval_data = None
    else:
        eval_data = None

    if config.prev_path != "none":
        model = GLiNER.from_pretrained(config.prev_path)
        model.config = config
    else:
        model = GLiNER(config)

    if torch.cuda.is_available():
        model = model.to('cuda')

    lr_encoder = float(config.lr_encoder)
    lr_others = float(config.lr_others)

    optimizer = torch.optim.AdamW([
        # encoder
        {'params': model.token_rep_layer.parameters(), 'lr': lr_encoder},
        {'params': model.rnn.parameters(), 'lr': lr_others},
        # projection layers
        {'params': model.span_rep_layer.parameters(), 'lr': lr_others},
        {'params': model.prompt_rep_layer.parameters(), 'lr': lr_others},
    ])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ######### DEBUG
    # data = data[:2]
    # ###############

    train(model, optimizer, data, eval_data=eval_data, num_steps=config.num_steps, eval_every=config.eval_every,
          log_dir=config.log_dir, wandb_log=args.wandb_log, warmup_ratio=config.warmup_ratio, train_batch_size=config.train_batch_size,
          device=device)
