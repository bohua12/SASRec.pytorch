import itertools
from utils import *
from utils.dataloader import data_partition, get_dataloader
from trainer import Trainer
import argparse
import numpy as np
import random
import torch
import os
import time


# Define hyperparameter grid search space
hidden_units = [64] 
lrs = [0.001, 0.0005, 0.0001] # Dont try 0.01
dropout_rates = [0.1, 0.2, 0.3, 0.4] # Sometimes even up to 0.9
weight_decays = [1e-4, 1e-3, 1e-2]

# Initialise Cmd Line args
parser = argparse.ArgumentParser()
#parser.add_argument('--dataset', required=True)
#parser.add_argument('--train_dir', required=True)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--seed', default=1111, type=int)

args = parser.parse_args()
print("Hyperparameter tuning")

# Tuning Loop
best_m = [-1,-1,999]
best_a = [-1,-1,999]
best_b = [-1,-1,999]
best_params = {}
num_tuning_epochs = 50
num_warmup_epochs = 30

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)

itr = 1
t_start = 0
t_end = 0
with open("tuning_2.txt", "w") as f:
    for hidden, lr, dropout, weight_decay in itertools.product(hidden_units, lrs, dropout_rates, weight_decays):
        start_time = time.time()  # Start timing the iteration

        # Update args values dynamically 
        args.hidden_units = hidden
        args.lr = lr
        args.dropout_rate = dropout
        args.weight_decay = weight_decay

        # LOAD MODEL
        trainer = Trainer(args)
        print(f"{itr}) Tuning for: hidden={hidden}, lr={lr}, dropout={dropout}, weight_decay={weight_decay}")
        f.write(f"{itr}) Tuning for: hidden={hidden}, lr={lr}, dropout={dropout}, weight_decay={weight_decay}")
        for epoch in range(1,num_tuning_epochs+1):
            epoch_train_loss = trainer.run_epoch(epoch)
            if epoch > num_warmup_epochs and epoch%10 == 0: # Epoch 50 onwards 
                m_valid, a_valid, b_valid = trainer.run_valid(epoch)
                print(f"Epoch {epoch}:")
                f.write(f"Epoch {epoch}:")
                f.write(f"M:({float(m_valid[0]):.5f}, {m_valid[1]:.5f}, {m_valid[2]:.5f})\n")
                f.write(f"A:({float(a_valid[0]):.5f}, {a_valid[1]:.5f}, {a_valid[2]:.5f})\n")
                f.write(f"B:({float(b_valid[0]):.5f}, {b_valid[1]:.5f}, {b_valid[2]:.5f})\n")
                if m_valid[2] < best_m[2]: # For now we do just on m
                    best_m = m_valid
                    #best_params = {"hidden_units": hidden, "lr": lr, "dropout_rate": dropout, "weight_decay": weight_decay}
                    f.write(f"===== ABOVE IS BEST SO FAR ====\n")
                    print(f"===== ABOVE IS BEST SO FAR ====\n")
        end_time = time.time()
        print(f"Time for iteration {itr}: {end_time - start_time:.2f} seconds")
        f.write(f"Time for iteration {itr}: {end_time - start_time:.2f} seconds\n")
        f.write("\n")
        f.flush()
        itr += 1
                                    
    f.write(f"Best OVERALL: {best_params}\n")
    print(f"M:({float(best_m[0]):.5f}, {best_m[1]:.5f}, {best_m[2]:.5f})")
    print(f"A:({float(best_a[0]):.5f}, {best_a[1]:.5f}, {best_a[2]:.5f})")
    print(f"B:({float(best_b[0]):.5f}, {best_b[1]:.5f}, {best_b[2]:.5f})")
    f.write(f"M:({float(best_m[0]):.5f}, {best_m[1]:.5f}, {best_m[2]:.5f})\n")
    f.write(f"A:({float(best_a[0]):.5f}, {best_a[1]:.5f}, {best_a[2]:.5f})\n")
    f.write(f"B:({float(best_b[0]):.5f}, {best_b[1]:.5f}, {best_b[2]:.5f})\n")
