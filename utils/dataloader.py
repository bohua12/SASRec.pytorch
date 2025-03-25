from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import numpy as np



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

def get_dataloader(user_train, usernum, itemnum, args):
    ds = SASRecDataset(user_train, usernum, itemnum, args.maxlen)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

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
    
# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t
