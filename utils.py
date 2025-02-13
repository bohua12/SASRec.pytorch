import sys
import copy
import random
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset


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
            # As long as nxt is a valid item (ie. not "0" padding), we can generate a negative item
            # By choosing any item not in ts
            if nxt != 0:
                neg[idx] = random_neq(1, self.itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        return uid, seq, pos, neg

"""Simple, split into train/test/valid"""
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')

    ## Store interaction in User Dict
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    ## IDGI why user_valid[] = [] so many times haha but other than that shd be ok
    ## For nFeedback < 3, all use to train
    ## >=3, last to test, 2ndLast to valid, rest to train
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

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
        ## TODO: My Unds: With that, Hit@10 and NDGC@10 can be evaluated (See how high the predicted item rank among 100 random items, compared to groun truth)
        ## TODO: Qn: How exactly does this work? 
        ## TODO: What exactly are we doing here (and the predictions below)
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
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

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
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user
