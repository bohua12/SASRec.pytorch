import itertools
import torch
from utils import *
from model import SASRec
from torch.utils.data import DataLoader
import argparse

# Define hyperparameter grid search space
hidden_units = [8, 16, 32, 64, 128 ,256]
lrs = [0.01, 0.001, 0.0001]
dropout_rates = [0.1, 0.2, 0.3, 0.4]
weight_decays = [1e-4, 1e-3, 1e-2]

# Initialise Cmd Line args
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--l2_emb', default=0.0, type=float)
args = parser.parse_args()

# Load dataset 
dataset = data_partition(args.dataset)
[user_train, user_valid, user_test, usernum, itemnum] = dataset
ds = SASRecDataset(user_train, usernum, itemnum, maxlen=args.maxlen)
dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

# Tuning Loop
best_val_metrics = [-1,-1,999]
best_params = {}
num_tuning_epochs = 5

with open("best_params.txt", "w") as f:
    for hidden, lr, dropout, weight_decay in itertools.product(hidden_units, lrs, dropout_rates, weight_decays):
        # Update args values dynamically 
        args.hidden_units = hidden
        args.lr = lr
        args.dropout_rate = dropout
        args.weight_decay = weight_decay

        # Initialize model
        model = SASRec(usernum, itemnum, args).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Quick training loop (train for a few mini-batches), breaks after 10 steps
        model.train()
        for epoch in range(num_tuning_epochs):
            for step, (u, seq, pos, neg) in enumerate(dataloader):
                u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()
                optimizer.zero_grad()
                pos_logits, neg_logits = model(u, seq, pos, neg)
                pos_labels, neg_labels = torch.ones_like(pos_logits), torch.zeros_like(neg_logits)
                loss = torch.nn.BCEWithLogitsLoss()(pos_logits, pos_labels) + torch.nn.BCEWithLogitsLoss()(neg_logits, neg_labels)
                loss.backward()
                optimizer.step()

        # Evaluate on validation set
        model.eval()
        val_metrics = evaluate_valid(model, dataset, args)
        print(f"Testing: hidden={hidden}, lr={lr}, dropout={dropout}, weight_decay={weight_decay}")
        f.write(f"Tested: hidden={hidden}, lr={lr}, dropout={dropout}, weight_decay={weight_decay}; {val_metrics}\n")
        
        # Save best hyperparameters
        ## TODO: Change to [0] and reverse < if want to do selection through NDCG !
        if val_metrics[2] < best_val_metrics[2]:
            best_val_metrics = val_metrics
            best_params = {"hidden_units": hidden, "lr": lr, "dropout_rate": dropout, "weight_decay": weight_decay}
            f.write(f"Best so far: {best_params}\n {val_metrics} \n")
            print(f"!!!!!!!! This itr is the best so far with these metrics {val_metrics} !!!!!!!!!\n")
    f.write(f"Best OVERALL: {best_params}\n {best_val_metrics} \n")

print("Best hyperparams:", best_params, best_val_metrics)