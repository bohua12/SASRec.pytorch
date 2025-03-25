import sys
import copy
import random
import numpy as np
from torch.utils.data import Dataset
import torch

""" User-item indexing"""
""" u2i_index: All items interacted by user u"""
""" i2u_index: Users who interact with item x"""
def build_index(dataset_name):
    """"""
    ui_mat = np.loadtxt('data/%s.txt' % dataset_name, dtype=np.int32)

    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()

    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])

    return u2i_index, i2u_index

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

""" Standard implementation of abstract class Dataset, to be instantiated and used with DataLoader class"""
class SASRecDataset(Dataset):
    def __init__(self, user_train, usernum, itemnum, maxlen):
        self.user_train = user_train
        self.usernum = usernum
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.users = list(user_train.keys())
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        # Gets uid of current indexed user
        uid = self.users[idx]
        # Initialise empty arrays of len(maxlen)
        seq = np.zeros([self.maxlen], dtype=np.int32)
        pos = np.zeros([self.maxlen], dtype=np.int32)
        neg = np.zeros([self.maxlen], dtype=np.int32) # Randomly sampled incorrect item (-ve sample)
        nxt = self.user_train[uid][-1]
        idx = self.maxlen - 1

        # Set of ALL items users[idx] have interacted with
        ts = set(self.user_train[uid])
        # Reason1 it is reversed: 0-Padding concept
        # Reason2 it is reversed: Fill in rightmost, latest ones first, and earliest interactions past maxLen are dropped
        for i in reversed(self.user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            # As long as "nxt" is a valid item (ie. not "0" padding), we can generate a -ve sample 
            # By choosing any item not in "ts"
            if nxt != 0:
                neg[idx] = random_neq(1, self.itemnum + 1, ts) 
            nxt = i
            idx -= 1
            if idx == -1:
                break

        return uid, seq, pos, neg



# TODO: merge evaluate functions for test and val set
# evaluate on test set
""" Evaluates Test Set"""
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    ## Normalized Discounted Cumulative Gain: Some eval metric
    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    ## Creates user Sequence
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        ## Generate -ve samples
        ## IIID: For each user, we randomly sample 100 -ve items, and rank these items (incl the predicted) with the ground-truth items
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        ## Actually predict
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            #print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user

# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    validation_loss = 0
    num_samples = 0

    users = range(1, usernum + 1)

    bce_criterion = torch.nn.BCEWithLogitsLoss()

    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: 
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: 
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        # Compute val loss
        # Had to unsqueeze to convert pos_logits to same shape as pos_label to calc loss!
        pos_logits = (-predictions[0]).unsqueeze(0)  # First item is the ground truth
        neg_logits = -predictions[1:]  # Remaining 100 are negative samples
        pos_label = torch.tensor([1.0], device=args.device)
        neg_labels = torch.zeros_like(neg_logits, device=args.device)

        loss = bce_criterion(pos_logits.to(args.device), pos_label)
        loss += bce_criterion(neg_logits.to(args.device), neg_labels)
        validation_loss += loss.item()
        num_samples += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            #print('.', end="")
            sys.stdout.flush()
    if num_samples > 0:
        avg_valid_loss = validation_loss / num_samples
    else:
        avg_valid_loss = float('inf')
    print(f"Validation Loss={validation_loss}, num_samples={num_samples}, avg_valid_loss={avg_valid_loss}")

    return NDCG / valid_user, HT / valid_user, avg_valid_loss

# For use per step (per epoch) in training loop (shifted code from main.py to here to clean up main.py)
def train_model(model, optimizer, bce_criterion, u, seq, pos, neg, args):
    model.train()
    
    u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()
    pos_logits, neg_logits = model(u, seq, pos, neg)
    
    # Labels: positive samples = 1, negative samples = 0
    pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)

    # Reset gradients
    optimizer.zero_grad()
    indices = np.where(pos != 0)

    # Compute BCE loss
    loss = bce_criterion(pos_logits[indices], pos_labels[indices])
    loss += bce_criterion(neg_logits[indices], neg_labels[indices])

    # L2 Regularization (Prevent Overfitting)
    for param in model.item_emb.parameters():
        loss += args.l2_emb * torch.norm(param)

    # Backpropagation
    loss.backward()
    optimizer.step()
    #print(f"Loss item: {loss.item()}")
    #print(f"Loss: {loss}")
    return loss
    
