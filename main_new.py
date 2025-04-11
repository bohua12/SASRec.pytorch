import os
import time
import torch
import argparse
from torch.utils.data import DataLoader
import random
from numpy import np

from model import SASRec, init_weights
from utils import *
from trainer import Trainer
def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

# Initialise Cmd Line args
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--hidden_units', default=32, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.1, type=float)
#parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
#parser.add_argument('--inference_only', default=False, type=str2bool)
#parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--weight_decay', default=1e-2, type=float)
parser.add_argument('--verification_frequency', default=5, type=int)
parser.add_argument('--early_stopping_patience', default=50, type=int)
parser.add_argument('--verbose', default=0, type=int)
parser.add_argument('--seed', default=1111, type=int)

# Create Training Directory
args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    trainer = Trainer(args)

    epoch_start_idx = 1
    best_val_ndcg, best_val_hr = 0.0, 0.0


    # SET UP LOGGING
    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    f.write('epoch (val_ndcg, val_hr, val_loss) (test_ndcg, test_hr)\n')

    ## TOTAL training time
    T = 0.0
    ## Training time since time of last eval (5 epochs)
    t0 = time.time()
    ## Keep track for early stoppage
    epochs_since_improvement = 0

    ## Actual Training
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        # HI CONSIDER ADDING scheduling here: lr_cur = trainer.optimizer.param_groups[0]["lr"]

        epoch_train_loss = trainer.run_epoch(epoch)

        ## Every 5 (by default) epoch, evaluate and save the model
        if epoch % args.verification_frequency == 0:

            ## Calculate total and current time (for this cuurrent batch of x epoch)
            t1 = time.time() - t0
            T += t1
            ## Eval test and validation performance
            t_test = trainer.run_test(epoch)
            t_valid = trainer.run_valid(epoch)
            print('epoch:%d, time taken: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f, validLoss: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                    % (epoch, t1, t_valid[0], t_valid[1], t_valid[2], t_test[0], t_test[1]))

            ## Save model only if either of the Validation Metrics improve (Not training Metrics)
            if t_valid[0] > best_val_ndcg or t_valid[1] > best_val_hr:
                best_val_ndcg = max(t_valid[0], best_val_ndcg)
                best_val_hr = max(t_valid[1], best_val_hr)
                folder = args.dataset + '_' + args.train_dir
                fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
                # MODEL SAVING: Next time? torch.save(model.state_dict(), os.path.join(folder, fname))
                f.write("[BEST]")

                epochs_since_improvement = 0
            else:
                epochs_since_improvement += args.verification_frequency
                print(f"{epochs_since_improvement} epochs since the last improvement in Validation NDCG/HR")
            # NEW SCHEDULER HERE!
            trainer.scheduler.step(t_valid[0])  # if you're monitoring NDCG
            ## Log results in log.txt
            f.write(str(epoch) + ' ' + str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()

            ## Activate early stoppage if patience is exceeded
            if epochs_since_improvement >= args.early_stopping_patience:
                print(f"Early stopping triggered at {epoch} epochs, after {epochs_since_improvement} epochs without improvement in Validaition NDCG/HR.")
                f.write(str(epoch) + "Early Stopping Triggered" + '\n')
                break
            ## Reset timer for next itr of x epochs
            t0 = time.time()
            
        # Close and save once reach desired number of epochs
        if epoch == args.num_epochs:
            folder = args.dataset + '_' + args.train_dir
            fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            # MODEL SAVING: Next time? torch.save(model.state_dict(), os.path.join(folder, fname))

    f.close()
    print("Done")
