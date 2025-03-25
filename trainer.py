from utils.utils import *
from utils.dataloader import data_partition, get_dataloader # STATE WHILE FN LATER!
from model import SASRec, init_weights
import torch
import os
""" Includes data loading"""
class Trainer(object):
    def __init__(self, args) -> None:
        self.args = args
        ### LOAD DATA ###
        print("Loading data...")
        # Split data into Train/Test/Valid
        [self.user_train, self.user_valid, self.user_test, self.n_users, self.n_items] = data_partition(args.dataset)
        # Get dataloader for training dataset
        self.dl = get_dataloader(self.user_train, self.n_users, self.n_items, args)
        print("Data loaded successfully!\n")

        ## CONSIDER ADDING calc avg seq len here!

        # LOAD MODEL
        print("Loading model...")
        self.model = SASRec(self.n_users, self.n_items, args).to(args.device)
        # adam moved from later line to here!
        self.adam_optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)
        self.bce_loss = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()

        for name, param in self.model.named_parameters():
            try:
                torch.nn.init.xavier_normal_(param.data)
            except:
                pass # just ignore those failed init layers

        init_weights(self.model)

        print("Model loaded successfully!\n")


    def run_epoch(self, i):
        self.model.train()
        self.adam_optimizer.zero_grad()
        epoch_loss = 0

        for step, (u, seq, pos, neg) in enumerate(self.dl):
            # TRAIN BATCH
            u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()
            pos_logits, neg_logits = self.model(u, seq, pos, neg) # Logits: Raw score before activation fn
            
            # CALCULATE LOSS (# Consider transfering below loss calculation to another function...)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.args.device), torch.zeros(neg_logits.shape, device=self.args.device) # Assign 1 to +ve items, 0 to -ve items
            indices = np.where(pos != 0) # "Give me positions in pos that are not padding (ie. actl items)" ; Can usethe same for pos and neg because both are generated from the same user-interaction sequence!
            loss = self.bce_loss(pos_logits[indices], pos_labels[indices])
            loss += self.bce_loss(neg_logits[indices], neg_labels[indices])

            # BACKWARD PROP
            loss.backward()
            self.adam_optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {i} trg loss:{epoch_loss / len(self.dl)}")
        return epoch_loss
    
    """
    Run validation on the current Model using the validation set.

    For each user:
    - Builds a sequence from train set.
    - Predicts the held-out validation item among 100 negative samples.
    - Computes ranking metrics (NDCG@10 and HitRate@10).
    - Computes Binary Cross Entropy loss for the positive and negative items.

    Args:
        i (int): Current epoch number (used for logging).

    Returns:
        tuple[int]: (NDCG@10, HitRate@10, average validation loss)
    """
    def run_valid(self, i):
        print(f"Validating on epoch {i}...", end="")

        self.model.eval()

        NDCG, HT = 0.0, 0.0
        valid_user, num_samples = 0.0, 0
        total_val_loss, val_loss = 0.0, 0.0

        users = range(1, self.n_users + 1)

        for u in users:
            if len(self.user_valid[u]) < 1: 
                continue
            # seq[] = train[u] (then we predict valid[u])
            seq = np.zeros([self.args.maxlen], dtype=np.int32)
            idx = self.args.maxlen - 1
            for i in reversed(self.user_train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1: break

            rated = set(self.user_train[u])
            rated.add(0)
            item_idx = [self.user_valid[u][0]]
            for _ in range(100):
                t = np.random.randint(1, self.n_items + 1)
                while t in rated: 
                    t = np.random.randint(1, self.n_items + 1)
                item_idx.append(t)

            predictions = -self.model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
            predictions = predictions[0]

            rank = predictions.argsort().argsort()[0].item()

            valid_user += 1

            # CALCULATE VAL LOSS
            # Had to unsqueeze to convert pos_logits to same shape as pos_label to calc loss!
            pos_logits = (-predictions[0]).unsqueeze(0)  # First item is the ground truth
            neg_logits = -predictions[1:]  # Remaining 100 are negative samples
            pos_label = torch.tensor([1.0], device=self.args.device)
            neg_labels = torch.zeros_like(neg_logits, device=self.args.device)

            loss = self.bce_loss(pos_logits.to(self.args.device), pos_label)
            loss += self.bce_loss(neg_logits.to(self.args.device), neg_labels)
            total_val_loss += loss.item()
            num_samples += 1

            if rank < 10:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1
            if valid_user % 100 == 0:
                print('.', end="")
                sys.stdout.flush()
                
        # Calculate validation loss
        if num_samples > 0:
            val_loss = total_val_loss / num_samples
            NDCG = NDCG / valid_user
            HT = HT / valid_user

        print(f"num_samples={num_samples}, avg_valid_loss={val_loss}")
        return NDCG, HT, val_loss

    def run_test(self, i):
        print(f"Testing on epoch {i}...", end="")

        self.model.eval()

        NDCG, HT = 0.0, 0.0
        valid_user= 0

        users = range(1, self.n_users + 1)

        # Reconstruct user sequence from train + valid
        # seq[] = train[u] + valid[u] (then we predict test[u])
        for u in users:
            if len(self.user_test[u]) < 1: 
                continue

            seq = np.zeros([self.args.maxlen], dtype=np.int32)
            idx = self.args.maxlen - 1
            seq[idx] = self.user_valid[u][0]
            idx -=1
            for i in reversed(self.user_train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1: break

            rated = set(self.user_train[u])
            rated.add(0)
            item_idx = [self.user_valid[u][0]]
            for _ in range(100):
                t = np.random.randint(1, self.n_items + 1)
                while t in rated: 
                    t = np.random.randint(1, self.n_items + 1)
                item_idx.append(t)

            predictions = -self.model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
            predictions = predictions[0]

            rank = predictions.argsort().argsort()[0].item()

            valid_user += 1

            if rank < 10:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1
            if valid_user % 100 == 0:
                print('.', end="")
                sys.stdout.flush()
                
        # Calculate validation loss
        if valid_user > 0:
            NDCG = NDCG / valid_user
            HT = HT / valid_user

        return NDCG, HT

