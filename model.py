import numpy as np
import torch


""" Why this one need create our own instead of just using torch lib?"""
class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device ## CUDA or CPU


        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        ## IIIA: Embedding Layer - Create item embedding (represent item)
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        ## IIIA: Embedding Layer-Positional Embedding - Create Positional Embedding (Because of nature of self-attention module)
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        ## IIIC: Stacking Self-Attention Blocks-Dropout - alleviate overfitting in Deep NN (randomly turn off neurons)
        ## Or does this refer to "Apply a drouput layer on embedding E" (E = Input Embedding Matrix)
        ## Set to 0.2 on default
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        ## Self-Attention layers
        ## IIIC: Stacking Self-Attention Blocks-Layer Normalization: Normalise input
        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()

        ## FF Layers
        ## IIIC: Stacking Self-Attention Blocks-Layer Normalization: Normalise input
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        ## IIID: Prediction Layer-Explicit User Modeling: Insert explicit user embedding at last layer (idgi, not even sure if its this one!)
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        ## IIIB: Self-Attention Block - Creating num_blocks number of self-attention blocks!
        ## TODO: QN: IS this the entire self-attention block? 
        for _ in range(args.num_blocks):

            ## IIIB: Self-Attention Block -Self-Attention Layer 
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            ## IIIB: Self-Attention Block-Point-Wise FFN & Normalise before FFN
            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    """ Generate user sequence embedding"""
    ## TODO: QN: Why can't just use torch.nn.embeddings but need item_emb?
    def log2feats(self, log_seqs): # TODO: fp64 and int64 as default in python, trim?
        ## Get embedding from item_emb (declared at __init__)
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        ## TODO:QNL Why we sqrt here? Ans: Dont want gradient to explode
        seqs *= self.item_emb.embedding_dim ** 0.5 
        ## Not sure whats poss for, ANS: its a numpy array
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        poss *= (log_seqs != 0)
        ## IIIA: Embedding Layer-Positional Embedding - Create Positional Embedding (Because of nature of self-attention module)
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        ## IIIC: Stacking Self-Attention Blocks-Dropout - alleviate overfitting in Deep NN (randomly turn off neurons)
        seqs = self.emb_dropout(seqs)

        ## Creates Mask (prevent peek into future) 
        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        ## Feed through all the attention blocks! (Created in __init__)
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # need_weights=False) this arg do not work?
            ## TODO: QN: What does this do?
            ## whats mha_outputs, multi head attention output>
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)
            ## FFN
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        ## IIID: Prediction Layer-Explicit User Modeling: Is it this? Insert explicit user embedding at last layer
        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    """ Seems like this function is never called anywehre..."""
    ### Seems like calculating +ve -ve sample scores
    ### self.model(data)  equals to self.model.forward(data). Special situation then use this fn
    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training        
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    """ Seems like this function is never called anywehre..."""
    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        ## different logits between calc matrix and loss. Need diff functions
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)
