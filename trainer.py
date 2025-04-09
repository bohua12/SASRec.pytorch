from utils.utils import *
from utils.dataloader import data_partition, data_partition, get_dataloader # STATE WHILE FN LATER!
from model_CDSR import CDSR, init_weights
import torch
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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
        ] = data_partition("abe", "abe_50_preprocessed.txt", args)

        # Get dataloader for training dataset
        #self.dl = get_dataloader(self.user_train, self.n_users, self.n_items, self.args)
        self.dl = get_dataloader(self.user_train_m, self.user_train_a, self.user_train_b, self.n_users, self.n_items_m, self.n_items_a, self.n_items_b, args)
        print("Data loaded successfully!\n")

        # LOAD MODEL
        print("Loading model...")
        #self.model = SASRec(self.n_users, self.n_items, args).to(args.device)
        self.model = CDSR(self.n_users, self.n_items_m, self.n_items_a, self.n_items_b, self.args).to(args.device)
        # adam moved from later line to here!
        self.adam_optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.adam_optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )

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

        for step, batch in enumerate(self.dl):
            current_lr = self.adam_optimizer.param_groups[0]["lr"]
            uid, seq_m, pos_m, neg_m, seq_a, pos_a, neg_a, seq_b, pos_b, neg_b = map(lambda x: x.to(self.args.device), batch)
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
                indices = (pos != 0)
                loss += self.bce_loss(pos_logits[indices], pos_labels[indices])
                loss += self.bce_loss(neg_logits[indices], neg_labels[indices])

            # BACKWARD PROP
            loss.backward()
            self.adam_optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {i} trg loss: {epoch_loss / len(self.dl):.4f} | LR: {current_lr:.6f}")
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
        print(f"Validating on epoch {i}")

        self.model.eval()

        NDCG_m, HT_m, NDCG_a, HT_a, NDCG_b, HT_b = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 
        invalid_m, invalid_a, invalid_b = 0, 0, 0
        valid_users_m, valid_users_b, valid_users_a = 0, 0, 0
        num_samples_m, num_samples_a, num_samples_b = 0, 0, 0
        val_loss_m, val_loss_a, val_loss_b = 0.0, 0.0, 0.0
        rank_m, rank_a, rank_b = 0,0,0


        users = range(self.n_users)

        for u in users:
            # seq[] = train[u] (then we predict valid[u])
            if len(self.user_valid_m[u]) > 0:
                seq_m = self.generate_validation_sequence(self.user_train_m[u], self.args.maxlen).to(self.args.device)
                

                rated = set(self.user_train_m[u])
                rated.add(0)
                item_idx = [self.user_valid_m[u][0]]
                for _ in range(100):
                    t = np.random.randint(1, self.n_items_m + 1)
                    while t in rated: 
                        t = np.random.randint(1, self.n_items_m + 1)
                    item_idx.append(t)

                pred_m = -self.model.predict(seq_m.unsqueeze(0), item_idx, 'm')

                ndgc, ht, rank = self.calc_metrics(pred_m)
                NDCG_m += ndgc
                HT_m += ht
                rank_m += rank
                valid_users_m += 1
                val_loss_m += self.calc_val_loss(pred_m)
                num_samples_m += 1
            else:
                invalid_m += 1

            if len(self.user_valid_a[u]) > 0:
                seq_a = self.generate_validation_sequence(self.user_train_a[u], self.args.maxlen).to(self.args.device)

                rated = set(self.user_train_a[u])
                rated.add(0)
                item_idx = [self.user_valid_a[u][0]]
                for _ in range(100):
                    t = np.random.randint(1, self.n_items_a + 1)
                    while t in rated: 
                        t = np.random.randint(1, self.n_items_a + 1)
                    item_idx.append(t)

                pred_a = -self.model.predict(seq_a.unsqueeze(0), item_idx, 'a')
                
                ndgc, ht, rank = self.calc_metrics(pred_a)
                NDCG_a += ndgc
                HT_a += ht
                rank_a += rank
                valid_users_a += 1
                val_loss_a += self.calc_val_loss(pred_a)
                num_samples_a += 1
            else:
                invalid_a += 1

            if len(self.user_valid_b[u]) > 0:
                seq_b = self.generate_validation_sequence(self.user_train_b[u], self.args.maxlen).to(self.args.device)

                rated = set(self.user_train_b[u])
                rated.add(0)
                item_idx = [self.user_valid_b[u][0]]
                for _ in range(100):
                    t = np.random.randint(1, self.n_items_b + 1)
                    while t in rated: 
                        t = np.random.randint(1, self.n_items_b + 1)
                    item_idx.append(t)

                pred_b = -self.model.predict(seq_b.unsqueeze(0), item_idx, 'b')
                
                ndgc, ht, rank = self.calc_metrics(pred_b)
                NDCG_b += ndgc
                HT_b += ht
                rank_b += rank
                valid_users_b += 1
                val_loss_b += self.calc_val_loss(pred_b)
                num_samples_b += 1
            else:
                invalid_b += 1

        if valid_users_m > 0:
            NDCG_m = NDCG_m / valid_users_m
            HT_m = HT_m / valid_users_m
            rank_m = rank_m / valid_users_m  # Add rank averaging for domain m
            val_loss_m = val_loss_m / valid_users_m

        if valid_users_a > 0:
            NDCG_a = NDCG_a / valid_users_a
            HT_a = HT_a / valid_users_a
            rank_a = rank_a / valid_users_a  # Add rank averaging for domain a
            val_loss_a = val_loss_a / valid_users_a

        if valid_users_b > 0:
            NDCG_b = NDCG_b / valid_users_b
            HT_b = HT_b / valid_users_b
            rank_b = rank_b / valid_users_b  # Add rank averaging for domain b
            val_loss_b = val_loss_b / valid_users_b
                
        print(f"M: {valid_users_m}/{invalid_m + valid_users_m} NDCG_m: {NDCG_m:.4f}, HT_m: {HT_m:.4f}, avgrank_m: {rank_m:.4f}, val_loss_m: {val_loss_m:.4f}")
        print(f"A: {valid_users_a}/{invalid_a + valid_users_a} NDCG_a: {NDCG_a:.4f}, HT_a: {HT_a:.4f}, avgrank_a: {rank_a:.4f}, val_loss_a: {val_loss_a:.4f}")
        print(f"B: {valid_users_b}/{invalid_b + valid_users_b} NDCG_b: {NDCG_b:.4f}, HT_b: {HT_b:.4f}, avgrank_b: {rank_b:.4f}, val_loss_b: {val_loss_b:.4f}")
        return NDCG_m, HT_m, val_loss_m

    def run_test(self, i):
        print(f"Testing on epoch {i}...")

        self.model.eval()

        NDCG_m, HT_m, NDCG_a, HT_a, NDCG_b, HT_b = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # filepath: c:\Users\leebo\SASRec.pytorch\trainer.py        valid_users_m, valid_users_a, valid_users_b= 0, 0, 0 
        invalid_m, invalid_a, invalid_b = 0, 0, 0
        valid_users_m, valid_users_b, valid_users_a = 0, 0, 0
        rank_m, rank_a, rank_b = 0,0,0

        users = range(self.n_users)

        for u in users:
            # if len(self.user_test_m[u]) < 1: 
            #     continue

            ## 1) GENERATE SEQUENCE
            # Reconstruct user sequence from train + valid for all 3 domains
            # seq[] = train[u] + valid[u] (then we predict test[u])


            ## 2) RATE ITEM
            # Rated and item_idx for m
            if len(self.user_test_m[u]) > 0 and len(self.user_valid_m[u]) > 0:
                seq_m = self.generate_test_sequence(self.user_test_m[u], self.user_train_m[u], self.user_valid_m[u], self.args.maxlen).to(self.args.device)
                rated_m = set(self.user_train_m[u])
                rated_m.add(0)
                item_idx_m = [self.user_test_m[u][0]]
                # QT: use his trick to avoid loop; this is v slow
                for _ in range(100): # Select 100 random item not in this domain
                    t = np.random.randint(1, self.n_items_m + 1)
                    while t in rated_m: 
                        t = np.random.randint(1, self.n_items_m + 1)
                    item_idx_m.append(t)
                pred_m = -self.model.predict(seq_m.unsqueeze(0), item_idx_m, 'm')
                delta_NDCG_m, delta_HT_m, delta_rank_m = self.calc_metrics(pred_m)
                NDCG_m += delta_NDCG_m
                HT_m += delta_HT_m
                rank_m += delta_rank_m
                valid_users_m += 1
                #print(f"NDCG_m: {NDCG_m:.4f}, HT_m: {HT_m:.4f}, Rank_m {rank_m}")
            else:
                invalid_m += 1

            # Rated and item_idx for a
            if len(self.user_test_a[u]) > 0 and len(self.user_valid_a[u]) > 0:
                seq_a = self.generate_test_sequence(self.user_test_a[u], self.user_train_a[u], self.user_valid_a[u], self.args.maxlen).to(self.args.device)
                rated_a = set(self.user_train_a[u])
                rated_a.add(0)
                item_idx_a = [self.user_test_a[u][0]]
                for _ in range(100):
                    t = np.random.randint(1, self.n_items_a + 1)
                    while t in rated_a: 
                        t = np.random.randint(1, self.n_items_a + 1)
                    item_idx_a.append(t)
                pred_a = -self.model.predict(seq_a.unsqueeze(0), item_idx_a, 'a')
                delta_NDCG_a, delta_HT_a, delta_rank_a = self.calc_metrics(pred_a)
                NDCG_a += delta_NDCG_a
                HT_a += delta_HT_a
                rank_a += delta_rank_a
                valid_users_a += 1
                #print(f"NDCG_a: {NDCG_a:.4f}, HT_a: {HT_a:.4f}, Rank_a {rank_a}")
            else:
                invalid_a += 1

            # Rated and item_idx for b
            if len(self.user_test_b[u]) > 0 and len(self.user_valid_b[u]) > 0:
                seq_b = self.generate_test_sequence(self.user_test_b[u], self.user_train_b[u], self.user_valid_b[u], self.args.maxlen).to(self.args.device)
                rated_b = set(self.user_train_b[u])
                rated_b.add(0)
                item_idx_b = [self.user_test_b[u][0]]
                for _ in range(100):
                    t = np.random.randint(1, self.n_items_b + 1)
                    while t in rated_b: 
                        t = np.random.randint(1, self.n_items_b + 1)
                    item_idx_b.append(t)
                pred_b = -self.model.predict(seq_b.unsqueeze(0), item_idx_b, 'b')
                delta_NDCG_b, delta_HT_b, delta_rank_b = self.calc_metrics(pred_b)
                NDCG_b += delta_NDCG_b
                HT_b += delta_HT_b
                rank_b += delta_rank_b
                valid_users_b += 1
                #print(f"NDCG_b: {NDCG_b:.4f}, HT_b: {HT_b:.4f}, Rank_b {rank_b}")
            else:
                invalid_b += 1

        if valid_users_m > 0:
            NDCG_m = NDCG_m / valid_users_m
            HT_m = HT_m / valid_users_m
            rank_m = rank_m / valid_users_m  # Add rank averaging for domain m

        if valid_users_a > 0:
            NDCG_a = NDCG_a / valid_users_a
            HT_a = HT_a / valid_users_a
            rank_a = rank_a / valid_users_a  # Add rank averaging for domain a

        if valid_users_b > 0:
            NDCG_b = NDCG_b / valid_users_b
            HT_b = HT_b / valid_users_b
            rank_b = rank_b / valid_users_b  # Add rank averaging for domain b
        print(f"M: {valid_users_m}/{invalid_m + valid_users_m} NDCG_m: {NDCG_m:.4f}, HT_m: {HT_m:.4f}, avgrank_m: {rank_m:.4f}")
        print(f"A: {valid_users_a}/{invalid_a + valid_users_a} NDCG_a: {NDCG_a:.4f}, HT_a: {HT_a:.4f}, avgrank_a: {rank_a:.4f}")
        print(f"B: {valid_users_b}/{invalid_b + valid_users_b} NDCG_b: {NDCG_b:.4f}, HT_b: {HT_b:.4f}, avgrank_b: {rank_b:.4f}")
        return NDCG_m, HT_m

    def calc_metrics(self, pred, target_rank = 10):
        pred = pred[0]  # remove batch dimension
        rank = pred.argsort().argsort()[0].item()
        NDCG = 0
        HT = 0
        if rank < target_rank:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        return NDCG, HT, rank
    
    def generate_test_sequence(self, user_test, user_train, user_valid, maxlen):
        if len(user_test) < 1:
            return False
        seq = torch.zeros(maxlen, dtype=torch.long)
        idx = maxlen - 1
        seq[idx] = user_valid[0]
        idx -= 1
        for i in reversed(user_train):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        return seq
    
    def generate_validation_sequence(self, user_train, maxlen):
        seq = torch.zeros(maxlen, dtype=torch.long)
        idx = maxlen - 1
        for i in reversed(user_train):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        return seq
    
    def calc_val_loss(self, predictions):
        predictions = predictions[0]
        pos_logits = (-predictions[0]).unsqueeze(0)  # First item is the ground truth
        neg_logits = -predictions[1:]  # Remaining 100 are negative samples
        pos_label = torch.tensor([1.0], device=self.args.device)
        neg_labels = torch.zeros_like(neg_logits, device=self.args.device)

        loss = self.bce_loss(pos_logits.to(self.args.device), pos_label)
        loss += self.bce_loss(neg_logits.to(self.args.device), neg_labels)
        
        return loss.item()