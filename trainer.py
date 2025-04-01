from utils.utils import *
from utils.dataloader import data_partition, data_partition_new, get_dataloader, get_dataloader_cdsr # STATE WHILE FN LATER!
from model_CDSR import CDSR, init_weights
#from model import SASRec, init_weights
import torch
import os

class Trainer(object):
    def __init__(self, args) -> None:
        self.args = args
        ### LOAD DATA ###
        print("Loading data...")
        # Split data into Train/Test/Valid
        #[self.user_train, self.user_valid, self.user_test, self.n_users, self.n_items] = data_partition(args.dataset)
        [   
        self.user_train_m, self.user_valid_m, self.user_test_m,
        self.user_train_a, self.user_valid_a, self.user_test_a,
        self.user_train_b, self.user_valid_b, self.user_test_b,
        self.n_users, self.n_items_m, self.n_items_a, self.n_items_b
        ] = data_partition_new("abe", "abe_50_preprocessed.txt", args)

        # Get dataloader for training dataset
        #self.dl = get_dataloader(self.user_train, self.n_users, self.n_items, self.args)
        self.dl = get_dataloader_cdsr(self.user_train_m, self.user_train_a, self.user_train_b, self.n_users, self.n_items_m, self.n_items_a, self.n_items_b, args)
        print("Data loaded successfully!\n")

        # LOAD MODEL
        print("Loading model...")
        #self.model = SASRec(self.n_users, self.n_items, args).to(args.device)
        self.model = CDSR(self.n_users, self.n_items_m, self.n_items_a, self.n_items_b, self.args).to(args.device)
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
        # for step, (u, seq, pos, neg) in enumerate(self.dl):
        #     u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()
        #     #print(seq)
        #     pos_logits, neg_logits = self.model(u, seq, pos, neg)

        for step, (uid, seq_m, pos_m, neg_m, seq_a, pos_a, neg_a, seq_b, pos_b, neg_b) in enumerate(self.dl):
            # TRAIN BATCH
            (pos_logits_m, neg_logits_m,
            pos_logits_a, neg_logits_a,
            pos_logits_b, neg_logits_b) = self.model(
                                                uid, 
                                                seq_m, pos_m, neg_m,
                                                seq_a, pos_a, neg_a,
                                                seq_b, pos_b, neg_b
                                                ) # Logits: Raw score before activation fn
            
            # CALCULATE LOSS (# Consider transfering below loss calculation to another function...)
            loss = 0
            for pos_logits, neg_logits, pos in [
                (pos_logits_m, neg_logits_m, pos_m),
                (pos_logits_a, neg_logits_a, pos_a),
                (pos_logits_b, neg_logits_b, pos_b)
            ]:
                pos_labels = torch.ones(pos_logits.shape, device=self.args.device)
                neg_labels = torch.zeros(neg_logits.shape, device=self.args.device)
                indices = np.where(np.array(pos) != 0)  # Only non-padding positions
                loss += self.bce_loss(pos_logits[indices], pos_labels[indices])
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

        users = range(self.n_users)

        for u in users:
            if len(self.user_valid_m[u]) < 1: 
                continue
            # seq[] = train[u] (then we predict valid[u])
            seq = np.zeros([self.args.maxlen], dtype=np.int32)
            idx = self.args.maxlen - 1
            for i in reversed(self.user_train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1: break

            seq_m = np.zeros([self.args.maxlen], dtype=np.int32)
            idx_m = self.args.maxlen - 1
            for i in reversed(self.user_train_m[u]):
                seq_m[idx_m] = i
                idx_m -= 1
                if idx_m == -1: break


            seq_a = np.zeros([self.args.maxlen], dtype=np.int32)
            idx_a = self.args.maxlen - 1
            for i in reversed(self.user_train_a[u]):
                seq_a[idx_a] = i
                idx_a -= 1
                if idx_a == -1: break



            seq_b = np.zeros([self.args.maxlen], dtype=np.int32)
            idx_b = self.args.maxlen - 1
            for i in reversed(self.user_train_b[u]):
                seq_b[idx_b] = i
                idx_b -= 1
                if idx_b == -1: break


            rated = set(self.user_train_m[u])
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

        users = range(self.n_users)
        print("len(self.user_test_m)", len(self.user_test_m))
        print("len(self.user_test_a)", len(self.user_test_a))
        print("len(self.user_test_b)", len(self.user_test_b))

        for u in users:
            if len(self.user_test_m[u]) < 1: 
                continue

            ## 1) GENERATE SEQUENCE
            # Reconstruct user sequence from train + valid for all 3 domains
            # seq[] = train[u] + valid[u] (then we predict test[u])
            seq_m = np.zeros([self.args.maxlen], dtype=np.int32)
            idx_m = self.args.maxlen - 1
            seq_m[idx_m] = self.user_valid_m[u][0]
            idx_m -= 1
            for i in reversed(self.user_train_m[u]):
                seq_m[idx_m] = i
                idx_m -= 1
                if idx_m == -1: break

            seq_a = np.zeros([self.args.maxlen], dtype=np.int32)
            idx_a = self.args.maxlen - 1
            if len(self.user_valid_a[u]) > 0:
                seq_a[idx_a] = self.user_valid_a[u][0]
                idx_a -= 1
            for i in reversed(self.user_train_a[u]):
                seq_a[idx_a] = i
                idx_a -= 1
                if idx_a == -1: break

            seq_b = np.zeros([self.args.maxlen], dtype=np.int32)
            idx_b = self.args.maxlen - 1
            if len(self.user_valid_b[u]) > 0:
                seq_b[idx_b] = self.user_valid_b[u][0]
                idx_b -= 1
            for i in reversed(self.user_train_b[u]):
                seq_b[idx_b] = i
                idx_b -= 1
                if idx_b == -1: break

            ## 2) RATE ITEM
            # Rated and item_idx for m
            rated_m = set(self.user_train_m[u])
            rated_m.add(0)
            item_idx_m = [self.user_test_m[u][0]]
            for _ in range(100): # Select 100 random item not in this domain
                t = np.random.randint(1, self.n_items_m + 1)
                while t in rated_m: 
                    t = np.random.randint(1, self.n_items_m + 1)
                item_idx_m.append(t)

            # Rated and item_idx for a
            rated_a = set(self.user_train_a[u])
            rated_a.add(0)
            item_idx_a = [self.user_test_a[u][0]]
            for _ in range(100):
                t = np.random.randint(1, self.n_items_a + 1)
                while t in rated_a: 
                    t = np.random.randint(1, self.n_items_a + 1)
                item_idx_a.append(t)

            # Rated and item_idx for b
            rated_b = set(self.user_train_b[u])
            rated_b.add(0)
            item_idx_b = [self.user_test_b[u][0]]
            for _ in range(100):
                t = np.random.randint(1, self.n_items_b + 1)
                while t in rated_b: 
                    t = np.random.randint(1, self.n_items_b + 1)
                item_idx_b.append(t)

            ## GENERATE RANK
            pred_m, pred_a, pred_b = -self.model.predict(u, np.array([seq_m]), np.array([seq_a]), np.array([seq_b]), item_idx_m, item_idx_a, item_idx_b)
            NDCG_m, HT_m, rank_m = self.calc_metrics(pred_m)
            NDCG_a, HT_a, rank_a = self.calc_metrics(pred_a)
            NDCG_b, HT_b, rank_b = self.calc_metrics(pred_b)

            print(f"NDCG_m: {NDCG_m:.4f}, HT_m: {HT_m:.4f}")
            print(f"NDCG_a: {NDCG_a:.4f}, HT_a: {HT_a:.4f}")
            print(f"NDCG_b: {NDCG_b:.4f}, HT_b: {HT_b:.4f}")
            
            print(f"Rank M: {rank_m}, Rank A: {rank_a}, Rank B: {rank_b}")
            valid_user += 1


            # if valid_user % 100 == 0:
            #     print('.', end="")
            #     sys.stdout.flush()

                
        # Calculate validation loss
        if valid_user > 0:
            NDCG = NDCG / valid_user
            HT = HT / valid_user

        return NDCG, HT

    def calc_metrics(prep, target_rank = 10):
        pred = pred[0]  # remove batch dimension
        rank = pred.argsort().argsort()[0].item()
        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        return NDCG, HT, rank