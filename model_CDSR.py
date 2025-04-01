import numpy as np
import torch


""" Called by __init__ in CDSR for nLayer times"""
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
    

class Encoder(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        ## Self-Attention layers
        self.attention_layernorms = torch.nn.ModuleList() # Normalise inputs before selfattention
        self.attention_layers = torch.nn.ModuleList() # Store multiple self-attention laters

        ## FFN Layers
        ## IIIC: Stacking Self-Attention Blocks-Layer Normalization: Normalise input
        self.forward_layernorms = torch.nn.ModuleList() # Store layer normalisation layers
        self.forward_layers = torch.nn.ModuleList() # Stores actual FFN

        ## IIID: Prediction Layer-Explicit User Modeling: Insert explicit user embedding at last layer 
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

    """ Pass in seqs and poss processed up till emb_dropout"""
    def forward(self, seqs, poss):
        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.args.device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # need_weights=False) this arg do not work?

            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)
            ## FFN
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        # Basically return log_feats?
        return self.last_layernorm(seqs)



# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py
class CDSR(torch.nn.Module):
    def __init__(self, user_num, item_num_m, item_num_b, item_num_a, args):
        super(CDSR, self).__init__()
        self.args = args
        self.user_num = user_num
        self.item_num = item_num_m

        ## IIIA: Embedding Layer - Create item embedding (represent item)
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)

        ## IIIA: Embedding Layer-Positional Embedding - Create Positional Embedding (Because of nature of self-attention module)
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)

        ## IIIC: Stacking Self-Attention Blocks-Dropout - alleviate overfitting in Deep NN (randomly turn off neurons)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate) # Set to 0.2 on default

        self.encoder_m = Encoder(args)
        self.encoder_a = Encoder(args)
        self.encoder_b = Encoder(args)

        if args.verbose:
            print("Positional embedding shape:", self.pos_emb.weight.shape)  # (batch_size, maxlen, hidden_units)
            print("Item embedding shape:", self.item_emb.weight.shape)  # (batch_size, maxlen, hidden_units)
            print("")
            print(f"# of attention blocks: {len(self.attention_layers)}")
            print(f"Structure of all attn layer: {self.attention_layers}")
            print(f"Structure of first attn layer: {self.attention_layers[0]}")
            print("")
            print(f"Structure of all FFN layer: {self.forward_layers}")
            print(f"Structure of first FFN layer: {self.forward_layers[0]}")


    """ Before encoding """
    def generate_input_embedding(self, log_seqs):
        #print(log_seqs)
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.args.device))
        seqs *= self.item_emb.embedding_dim ** 0.5  # Sqrt to prevent Gradient Explosion

        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        poss *= (log_seqs != 0)

        seqs += self.pos_emb(torch.LongTensor(poss).to(self.args.device))
        seqs = self.emb_dropout(seqs)

        return seqs, poss


    ### self.model(data)  equals to self.model.forward(data). Special situation then use this fn
    def forward(self, uid, seq_m, pos_m, neg_m, seq_a, pos_a, neg_a, seq_b, pos_b, neg_b):

        log_feats_m = self.encoder_m(*self.generate_input_embedding(seq_m))
        log_feats_a = self.encoder_a(*self.generate_input_embedding(seq_a))
        log_feats_b = self.encoder_b(*self.generate_input_embedding(seq_b))

        ## item_emb obj shd be reused 
        pos_embs_m = self.item_emb(torch.LongTensor(pos_m).to(self.args.device))
        neg_embs_m = self.item_emb(torch.LongTensor(neg_m).to(self.args.device))
        
        pos_embs_a = self.item_emb(torch.LongTensor(pos_a).to(self.args.device))
        neg_embs_a = self.item_emb(torch.LongTensor(neg_a).to(self.args.device))

        pos_embs_b = self.item_emb(torch.LongTensor(pos_b).to(self.args.device))
        neg_embs_b = self.item_emb(torch.LongTensor(neg_b).to(self.args.device))

        # Compute logits
        pos_logits_m = (log_feats_m * pos_embs_m).sum(dim=-1)
        neg_logits_m = (log_feats_m * neg_embs_m).sum(dim=-1)

        pos_logits_a = (log_feats_a * pos_embs_a).sum(dim=-1)
        neg_logits_a = (log_feats_a * neg_embs_a).sum(dim=-1)

        pos_logits_b = (log_feats_b * pos_embs_b).sum(dim=-1)
        neg_logits_b = (log_feats_b * neg_embs_b).sum(dim=-1)


        return (pos_logits_m, neg_logits_m,
                pos_logits_a, neg_logits_a,
                pos_logits_b, neg_logits_b)

    """ Seems like this function is never called anywehre..."""
    def predict(self, user_ids, seq_m, seq_a, seq_b, item_idx_m, item_idx_a, item_idx_b): # for inference

        log_feats_m = self.encoder_m(*self.generate_input_embedding(seq_m))
        log_feats_a = self.encoder_a(*self.generate_input_embedding(seq_a))
        log_feats_b = self.encoder_b(*self.generate_input_embedding(seq_b))

        final_feat_m = log_feats_m[:, -1, :]
        final_feat_a = log_feats_a[:, -1, :]
        final_feat_b = log_feats_b[:, -1, :] 

        # Get item embeddings
        item_embs_m = self.item_emb(torch.LongTensor(item_idx_m).to(self.args.device))  # (batch_size, item_count, dim)
        item_embs_a = self.item_emb(torch.LongTensor(item_idx_a).to(self.args.device))  # (batch_size, item_count, dim)
        item_embs_b = self.item_emb(torch.LongTensor(item_idx_b).to(self.args.device))  # (batch_size, item_count, dim)

        # Predict scores
        logits_m = item_embs_m.matmul(final_feat_m.unsqueeze(-1)).squeeze(-1)  # (batch_size, item_count)
        logits_a = item_embs_a.matmul(final_feat_a.unsqueeze(-1)).squeeze(-1)  # (batch_size, item_count)
        logits_b = item_embs_b.matmul(final_feat_b.unsqueeze(-1)).squeeze(-1)  # (batch_size, item_count)

        return logits_m,logits_a,logits_b


def init_weights(model):
    """
    Initializes model weights using appropriate distributions.
    - Linear layers: Normal(mean=0, std=0.02), bias=0
    - Embeddings: Normal(mean=0, std=0.02)
    - LayerNorm: Weight=1, Bias=0
    """
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias.data)

        elif isinstance(m, (torch.nn.Embedding, torch.nn.Parameter)):
            torch.nn.init.normal_(m.weight.data, mean=0.0, std=0.02)

        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.ones_(m.weight.data)
            torch.nn.init.zeros_(m.bias.data)
