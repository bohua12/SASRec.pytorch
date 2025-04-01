from os.path import join
import json
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

def collate_identity(batch):
    # Allows us to return np.array for __getitem__, instead of tensor
    return tuple(np.stack(t) for t in zip(*batch))


def data_partition(fname, fraw, args):
    with open(join("data", fname, 'map_item.txt'), 'r') as f:
        map_i = json.load(f)
        list_dm = np.array(list(map_i.values()))[:, 1] # Slice out just domain col
        n_items_a = np.sum(list_dm == 0) # Count how many itrems in dom a
        n_items_b = np.sum(list_dm == 1)

    User = defaultdict(list)
    user_train_a, user_valid_a, user_test_a = {}, {}, {}
    user_train_b, user_valid_b, user_test_b = {}, {}, {}
    user_train_m, user_valid_m, user_test_m = {}, {}, {}
    
    with open(join('data', fname, fraw), 'r', encoding='utf-8') as f:
        for line in f:
            seq = []
            line = line.strip().split(' ')
            u = int(line[0])
            for ui in line[1:][-args.maxlen:]:
                User[u].append(int(ui.split('|')[0]))    
        for user in User:
            nfeedback = len(User[user])
            if nfeedback < 3:
                user_train_m[user] = User[user]
                user_valid_m[user] = []
                user_test_m[user] = []

                user_train_a[user] = [i for i in User[user] if i < n_items_a]
                user_train_b[user] = [i for i in User[user] if i >= n_items_a]
                user_valid_a[user] = []
                user_valid_b[user] = []
                user_test_a[user] = []
                user_test_b[user] = []
            else:
                user_train_m[user] = User[user][:-2]
                user_valid_m[user] = [User[user][-2]]
                user_test_m[user] = [User[user][-1]]

                user_train_a[user] = [i for i in User[user][:-2] if i < n_items_a]
                user_train_b[user] = [i for i in User[user][:-2] if i >= n_items_a]

                user_valid_a[user] = [User[user][-2]] if User[user][-2] < n_items_a else []
                user_valid_b[user] = [User[user][-2]] if User[user][-2] >= n_items_a else []

                user_test_a[user] = [User[user][-1]] if User[user][-1] < n_items_a else []
                user_test_b[user] = [User[user][-1]] if User[user][-1] >= n_items_a else []

    print(f"user_train_a: {len(user_train_a)}")
    print(f"user_valid_a: {len(user_valid_a)}")
    print(f"user_test_a: {len(user_test_a)}")

    print(f"user_train_b: {len(user_train_b)}")
    print(f"user_valid_b: {len(user_valid_b)}")
    print(f"user_test_b: {len(user_test_b)}")

    print(f"user_train_m: {len(user_train_m)}")
    print(f"user_valid_m: {len(user_valid_m)}")
    print(f"user_test_m: {len(user_test_m)}")

    n_users = len(user_train_m)
    print(n_users)
    
    n_items_m = n_items_a + n_items_b

    return (
        user_train_m, user_valid_m, user_test_m,
        user_train_a, user_valid_a, user_test_a,
        user_train_b, user_valid_b, user_test_b,
        n_users, n_items_m, n_items_a, n_items_b
    )

"""Simple, split into train/test/valid"""
def data_partition_old(fname):
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

def get_dataloader_old(user_train, usernum, itemnum, args):
    ds = SASRecDataset(user_train, usernum, itemnum, args.maxlen)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

def get_dataloader(train_m, train_a, train_b, usernum, itemnum_m, itemnum_a, itemnum_b, args):
    ds = CDSRDataset(train_m, train_a, train_b, usernum, itemnum_m, itemnum_a, itemnum_b, args)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_identity)

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
            if nxt != 0:
                neg[idx] = random_neq(1, self.itemnum + 1, ts) #random_neq generates a random -ve sample user never inr with
            nxt = i
            idx -= 1
            if idx == -1:
                break

        return uid, seq, pos, neg
    

class CDSRDataset(Dataset):
    def __init__(self, train_m, train_a, train_b, usernum, itemnum_m, itemnum_a, itemnum_b, args):
        self.train_m = train_m
        self.train_a = train_a
        self.train_b = train_b
        self.usernum = usernum
        self.itemnum_m = itemnum_m
        self.itemnum_a = itemnum_a
        self.itemnum_b = itemnum_b
        self.maxlen = args.maxlen
        self.users = list(train_m.keys())  # all 3 dataset will contain all users (handled in data_partition_new)
    
    def __len__(self):
        return len(self.users)
    
    """
    Constructs input, positive, and negative sequences for training with legnth maxlen

    Given a user's interaction sequence:
    - `seq_arr` stores the historical items (input sequence).
    - `pos_arr` stores the ground truth next item for each timestep.
    - `neg_arr` stores randomly sampled negative items that the user has not interacted with.

    This prepares data for SASRec training, where at each step the model learns to predict the next item,
    while distinguishing it from negative samples.

    Args:
        seq (list[int]): User's sequence of interacted items.
        itemnum (int): Total number of items in this domain (used for negative sampling range).

    Returns:
        tuple: (seq_arr, pos_arr, neg_arr) 
    """
    def build_sequence(self, seq, itemnum):
        ts = set(seq) 
        seq_arr = np.zeros([self.maxlen], dtype=np.int32)
        pos_arr = np.zeros([self.maxlen], dtype=np.int32)
        neg_arr = np.zeros([self.maxlen], dtype=np.int32)
        nxt = seq[-1]
        idx = self.maxlen - 1
        for i in reversed(seq[:-1]):
            seq_arr[idx] = i
            pos_arr[idx] = nxt
            if nxt != 0: # ie. only generate -ve sample if next item is valid
                neg_arr[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break
        return seq_arr, pos_arr, neg_arr
    
    def __getitem__(self, idx):
        # Gets uid of all users in processed dataset
        uid = self.users[idx]
        
        # get sequuence
        seq_m, pos_m, neg_m = self.build_sequence(self.train_m[uid], self.itemnum_m)
        seq_a, pos_a, neg_a = self.build_sequence(self.train_a[uid], self.itemnum_a)
        seq_b, pos_b, neg_b = self.build_sequence(self.train_b[uid], self.itemnum_b)

        return (
            uid,
            seq_m,
            pos_m,
            neg_m,
            seq_a,
            pos_a,
            neg_a,
            seq_b,
            pos_b,
            neg_b
        )

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t
