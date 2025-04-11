import itertools
from utils import *
from utils.dataloader import data_partition, get_dataloader
from trainer import Trainer
import argparse
import numpy as np

# Define hyperparameter grid search space
hidden_units = [32, 64, 128] 
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
args = parser.parse_args()
print("Hyperparameter tuning")
# # Load dataset 
# print("Loading data...")
# args.verbose=0

# # Load and Split data into Train/Test/Valid
# [
# user_train_m, user_valid_m, user_test_m,
# user_train_a, user_valid_a, user_test_a,
# user_train_b, user_valid_b, user_test_b,
# n_users, n_items_m, n_items_a, n_items_b
# ] = data_partition("abe", "abe_50_preprocessed.txt", args)

# # Get dataloader for training dataset
# dl = get_dataloader(user_train_m, user_train_a, user_train_b, n_users, n_items_m, n_items_a, n_items_b, args)
# print("Data loaded successfully!\n")

# Tuning Loop
best_m = [-1,-1,999]
best_a = [-1,-1,999]
best_b = [-1,-1,999]
best_params = {}
num_tuning_epochs = 50
num_warmup_epochs = 30
itr = 1

with open("best_params_100epochs.txt", "w") as f:
    for hidden, lr, dropout, weight_decay in itertools.product(hidden_units, lrs, dropout_rates, weight_decays):
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
                print(f"M:({float(m_valid[0]):.5f}, {m_valid[1]:.5f}, {m_valid[2]:.5f})\n")
                print(f"A:({float(a_valid[0]):.5f}, {a_valid[1]:.5f}, {a_valid[2]:.5f})\n")
                print(f"B:({float(b_valid[0]):.5f}, {b_valid[1]:.5f}, {b_valid[2]:.5f})\n")
                f.write(f"M:({float(m_valid[0]):.5f}, {m_valid[1]:.5f}, {m_valid[2]:.5f})\n")
                f.write(f"A:({float(a_valid[0]):.5f}, {a_valid[1]:.5f}, {a_valid[2]:.5f})\n")
                f.write(f"B:({float(b_valid[0]):.5f}, {b_valid[1]:.5f}, {b_valid[2]:.5f})\n")
                if m_valid[2] < best_m[2]: # For now we do just on m
                    best_m = m_valid
                    #best_params = {"hidden_units": hidden, "lr": lr, "dropout_rate": dropout, "weight_decay": weight_decay}
                    f.write(f"===== ABOVE IS BEST SO FAR ====\n")
                    print(f"===== ABOVE IS BEST SO FAR ====\n")
        f.write("\n")
        itr += 1
                                    
    f.write(f"Best OVERALL: {best_params}\n")
    print(f"M:({float(best_m[0]):.5f}, {best_m[1]:.5f}, {best_m[2]:.5f})\n")
    print(f"A:({float(best_a[0]):.5f}, {best_a[1]:.5f}, {best_a[2]:.5f})\n")
    print(f"B:({float(best_b[0]):.5f}, {best_b[1]:.5f}, {best_b[2]:.5f})\n")
    f.write(f"M:({float(best_m[0]):.5f}, {best_m[1]:.5f}, {best_m[2]:.5f})\n")
    f.write(f"A:({float(best_a[0]):.5f}, {best_a[1]:.5f}, {best_a[2]:.5f})\n")
    f.write(f"B:({float(best_b[0]):.5f}, {best_b[1]:.5f}, {best_b[2]:.5f})\n")
