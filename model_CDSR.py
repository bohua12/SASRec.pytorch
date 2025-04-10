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
    def __init__(self, user_num, item_num_m, item_num_a, item_num_b, args):
        super(CDSR, self).__init__()
        self.args = args
        self.user_num = user_num
        self.item_num = item_num_m
        self.item_num_a = item_num_a

        # === SHARED EMBEDDINGS ===
        ## IIIA: Embedding Layer - Create item embedding (represent item)
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        ## IIIA: Embedding Layer-Positional Embedding - Create Positional Embedding (Because of nature of self-attention module)
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        ## IIIC: Stacking Self-Attention Blocks-Dropout - alleviate overfitting in Deep NN (randomly turn off neurons)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate) # Set to 0.2 on default

        # === DOMAIN SPECIFIC ENCODERS ===
        self.encoder_m = Encoder(args)
        self.encoder_a = Encoder(args)
        self.encoder_b = Encoder(args)

        # === DOMAIN SPECIFIC LINEAR LAYERS === 
        self.lin_m = torch.nn.Linear(args.hidden_units, item_num_m + 1)
        self.lin_a = torch.nn.Linear(args.hidden_units, item_num_a + 1)
        self.lin_b = torch.nn.Linear(args.hidden_units, item_num_b + 1)

    """ Before encoding, trains the embedding """
    def generate_input_embedding(self, log_seqs):
        #print(log_seqs)
        seqs = self.item_emb(log_seqs.to(self.args.device))
        seqs *= self.item_emb.embedding_dim ** 0.5  # Sqrt to prevent Gradient Explosion

        # poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        # poss *= (log_seqs != 0)

        poss = torch.arange(1, log_seqs.shape[1] + 1).repeat(log_seqs.shape[0], 1)
        poss = poss.to(log_seqs.device)
        poss *= (log_seqs != 0)

        seqs += self.pos_emb(poss)
        seqs = self.emb_dropout(seqs)

        return seqs, poss
    
    ### self.model(data)  equals to self.model.forward(data). Special situation then use this fn
    def forward(self, uid, seq_m, pos_m, neg_m, seq_a, pos_a, neg_a, seq_b, pos_b, neg_b):

        log_feats_m = self.encoder_m(*self.generate_input_embedding(seq_m)) # Trains the embedding
        log_feats_a = self.encoder_a(*self.generate_input_embedding(seq_a))
        log_feats_b = self.encoder_b(*self.generate_input_embedding(seq_b))

        # Compute domain-specific scores in linear layer
        score_m = self.lin_m(log_feats_m)
        score_a = self.lin_a(log_feats_a)
        score_b = self.lin_b(log_feats_b)

        pos_b = torch.where(pos_b > 0, pos_b - self.item_num_a, torch.zeros_like(pos_b))
        neg_b = torch.where(neg_b > 0, neg_b - self.item_num_a, torch.zeros_like(neg_b))

        # Get positive/negative logit using gather
        pos_logits_m = torch.gather(score_m, dim=-1, index=pos_m.long().unsqueeze(-1)).squeeze(-1)
        neg_logits_m = torch.gather(score_m, dim=-1, index=neg_m.long().unsqueeze(-1)).squeeze(-1)
        pos_logits_a = torch.gather(score_a, dim=-1, index=pos_a.long().unsqueeze(-1)).squeeze(-1)
        neg_logits_a = torch.gather(score_a, dim=-1, index=neg_a.long().unsqueeze(-1)).squeeze(-1)
        pos_logits_b = torch.gather(score_b, dim=-1, index=pos_b.long().unsqueeze(-1)).squeeze(-1)
        neg_logits_b = torch.gather(score_b, dim=-1, index=neg_b.long().unsqueeze(-1)).squeeze(-1)

        return (pos_logits_m, neg_logits_m,
                pos_logits_a, neg_logits_a,
                pos_logits_b, neg_logits_b)

    def predict(self, seq, item_idx, domain):
        item_idx = torch.LongTensor(item_idx).to(self.args.device)
        seqs, poss = self.generate_input_embedding(seq)

        shared_feats = self.encoder_m(seqs, poss)[:, -1]  # assuming encoder_m is shared

        if domain == "a":
            specific_feats = self.encoder_a(seqs, poss)[:, -1]
            combined = shared_feats + specific_feats
            scores = self.lin_a(combined)
        elif domain == "b":
            specific_feats = self.encoder_b(seqs, poss)[:, -1]
            combined = shared_feats + specific_feats
            scores = self.lin_b(combined)
        else:
            scores = self.lin_m(shared_feats)  # shared-only

        return scores[:, item_idx]

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
