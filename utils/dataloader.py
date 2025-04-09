from os.path import join
import json
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

def collate_identity(batch):
    # Allows us to return np.array for __getitem__, instead of tensor (not used anymore)
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
    
    # NOTE: This runs on assumption that for every user, it has atleast 3 item per domain. 
    with open(join('data', fname, fraw), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            u = int(line[0])
            for ui in line[1:][-args.maxlen:]:
                User[u].append(int(ui.split('|')[0]))    
        for user in User:
            user_tensor = torch.LongTensor(User[user])
            a_items = user_tensor[user_tensor < n_items_a]
            b_items = user_tensor[user_tensor >= n_items_a]
            if len(a_items) < 3 or len(b_items) < 3:
                # TODO: Initialise empty tensor for that user. But for now ignore as it will have downstream impact
                print("not suppoused to happen")
            else:
                user_train_m[user] = user_tensor[:-2]
                user_valid_m[user] = user_tensor[-2:-1]
                user_test_m[user] = user_tensor[-1:]

                user_train_a[user] = a_items[:-2]
                user_valid_a[user] = a_items[-2:-1]
                user_test_a[user] = a_items[-1:]

                user_train_b[user] = b_items[:-2]
                user_valid_b[user] = b_items[-2:-1]
                user_test_b[user] = b_items[-1:]
                # print(f"User {user}: Train_M={len(user_train_m[user])}, Valid_M={len(user_valid_m[user])}, Test_M={len(user_test_m[user])}")
                # print(f"User {user}: Train_A={len(user_train_a[user])}, Valid_A={len(user_valid_a[user])}, Test_A={len(user_test_a[user])}")
                # print(f"User {user}: Train_B={len(user_train_b[user])}, Valid_B={len(user_valid_b[user])}, Test_B={len(user_test_b[user])}")
                # print("")

    n_users = len(user_train_m)
    print(n_users)
    n_items_m = n_items_a + n_items_b
    print("n_items_m", n_items_m)
    print("n_items_a", n_items_a)
    print("n_items_b", n_items_b)

    return (
        user_train_m, user_valid_m, user_test_m,
        user_train_a, user_valid_a, user_test_a,
        user_train_b, user_valid_b, user_test_b,
        n_users, n_items_m, n_items_a, n_items_b
    )

def get_dataloader(train_m, train_a, train_b, usernum, itemnum_m, itemnum_a, itemnum_b, args):
    ds = CDSRDataset(train_m, train_a, train_b, usernum, itemnum_m, itemnum_a, itemnum_b, args)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0,)    

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
        seq_arr = torch.zeros(self.maxlen, dtype=torch.long)
        pos_arr = torch.zeros(self.maxlen, dtype=torch.long)
        neg_arr = torch.zeros(self.maxlen, dtype=torch.long)
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
        neg_b += self.itemnum_a  # shift B-domain neg samples to global item ID space


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
