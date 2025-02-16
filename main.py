import os
import time
import torch
import argparse
from torch.utils.data import DataLoader

from model import SASRec
from utils import *

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

# Initialise Cmd Line args
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--weight_decay', default=1e-3, type=float)
parser.add_argument('--verification_frequency', default=5, type=int)


# Create Training Directory
args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':

    ## Generate User-Item mapping
    u2i_index, i2u_index = build_index(args.dataset)
    
    ## data_partition (in utils.py) splits dataset into train,valid,test interactions, 
    ## and also the number of user and item we're concerned
    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    num_batch = (len(user_train) - 1) // args.batch_size + 1

    ## Calculae Avg sequence length
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    
    ## Set up files for logging
    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    f.write('epoch (val_ndcg, val_hr) (test_ndcg, test_hr)\n')
    
    # Set up dataset and dataloader
    ds = SASRecDataset(user_train, usernum, itemnum, args.maxlen)
    dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    ## Generate instance of the SASRec model
    model = SASRec(usernum, itemnum, args).to(args.device) # no ReLU activation in original SASRec implementation?

    ## CXavier initialition for model weights 
    ## TODO: Find out wth is this. Is it part of the model. Just keep it here low priority for understanding; it just affect performance
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers

    ## Zeroes out embeddings for padding tokens (??)
    ## Is it the padding u told me about? but idg how this does that
    ## Ans: This is wrong!! Shd be initialising with 
    ## All zero will slow down in beginiing (See qingtian's init code)
    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0

    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)
    
    model.train() # enable model training
    
    epoch_start_idx = 1

    ## Load pre-trained model (If provided.) By default no
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()
            
    ## If arg included inference_only as True
    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))
    
    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...

    ## (Mentioned in III.E:Network Tuning)
    ## Use Binary Cross Entropy as objective function 
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    ## (Mentioned in III.E:Network Tuning)

    ## Use Adam Optimizer with weight decay (updated implementation)
    adam_optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)

    ## Use adam optimizer without weight decay (original implementation)
    #adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    best_val_ndcg, best_val_hr = 0.0, 0.0
    # best_test_ndcg, best_test_hr = 0.0, 0.0

    ## TOTAL training time
    T = 0.0
    ## Training time since time of last eval (5 epochs)
    t0 = time.time()

    ## Actual Training
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        ## TODO: qn: Was wodnering why not just cut off code at ~line 100 if we are just gunna break everything here?
        if args.inference_only: break # just to decrease identition
        print("========== Epoch ", epoch, " ==========")

        # Fetch data from DataLoader (user ID, interaction Sequence, +ve Samples, -ve Samples)
        for step, (u, seq, pos, neg) in enumerate(dataloader):
            u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()
            # pos_logits: Likelihood +ve samples; derieved from dot product 
            pos_logits, neg_logits = model(u, seq, pos, neg)
            # pos_labels: correct item high logit; neg_labels: incorrect item low logit
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)

            ## Ok use the adam for network tuning, tune hyperparams
            adam_optimizer.zero_grad() ## Set all gradients to 0 before backprop
            indices = np.where(pos != 0)

            ## Binary Cross Entropy loss for networking tuning. (Use BCE because this is a binary classification task)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            ## L2 regularization (prevent overfitting) -> Value is hyperparam
            for param in model.item_emb.parameters(): 
                loss += args.l2_emb * torch.norm(param)

            ## Backward propagation compute gradient of loss (adjust weights of previous layers)
            ## Note: backward() is running in C++ and CUDA so i cannot click inside; Idea is based on dynamic graph. Only very high level PHD need to unds
            loss.backward()

            adam_optimizer.step()
            print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs

        ## Every 5 (by default) epoch, evaluate and save the model
        if epoch % args.verification_frequency == 0:
            ## Toggle model to eval mode (part of pytorch implementation)
            model.eval()

            ## Calculate total and current time (for this cuurrent batch of x epoch)
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            ## Eval test and validation performance
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                    % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

            ## Save model only if either of the Validation Metrics improve (Not training Metrics)
            ## Reason being, Training Loss is more for evaluating hyperparam settings
            # TODO: Hyperparameter to change metrics used to evaluate model saving
            if t_valid[0] > best_val_ndcg or t_valid[1] > best_val_hr:
                best_val_ndcg = max(t_valid[0], best_val_ndcg)
                best_val_hr = max(t_valid[1], best_val_hr)
                # best_test_ndcg = max(t_test[0], best_test_ndcg)
                # best_test_hr = max(t_test[1], best_test_hr)
                folder = args.dataset + '_' + args.train_dir
                fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
                torch.save(model.state_dict(), os.path.join(folder, fname))

            ## Log results in log.txt
            f.write(str(epoch) + ' ' + str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            ## Reset timer for next itr of x epochs
            t0 = time.time()
            ## Toggle model back to Training mode (part of pytorch module.py)
            model.train()
            
        # Close and save once reach desired number of epochs
        if epoch == args.num_epochs:
            folder = args.dataset + '_' + args.train_dir
            fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            torch.save(model.state_dict(), os.path.join(folder, fname))

    f.close()
    print("Done")
