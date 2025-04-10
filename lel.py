def forward(self, uid, seq_m, pos_m, neg_m, seq_a, pos_a, neg_a, seq_b, pos_b, neg_b):
    log_feats_m = self.encoder_m(*self.generate_input_embedding(seq_m))
    log_feats_a = self.encoder_a(*self.generate_input_embedding(seq_a))
    log_feats_b = self.encoder_b(*self.generate_input_embedding(seq_b))

    # === Use last timestep of sequence ===
    last_feat_m = log_feats_m[:, -1]
    last_feat_a = log_feats_a[:, -1]
    last_feat_b = log_feats_b[:, -1]

    # === Compute logits via linear scoring layer ===
    pos_logits_m = self.lin_m(last_feat_m).gather(1, torch.LongTensor(pos_m).to(self.args.device).unsqueeze(1)).squeeze(1)
    neg_logits_m = self.lin_m(last_feat_m).gather(1, torch.LongTensor(neg_m).to(self.args.device).unsqueeze(1)).squeeze(1)

    pos_logits_a = self.lin_a(last_feat_a).gather(1, torch.LongTensor(pos_a).to(self.args.device).unsqueeze(1)).squeeze(1)
    neg_logits_a = self.lin_a(last_feat_a).gather(1, torch.LongTensor(neg_a).to(self.args.device).unsqueeze(1)).squeeze(1)

    pos_logits_b = self.lin_b(last_feat_b).gather(1, torch.LongTensor(pos_b).to(self.args.device).unsqueeze(1)).squeeze(1)
    neg_logits_b = self.lin_b(last_feat_b).gather(1, torch.LongTensor(neg_b).to(self.args.device).unsqueeze(1)).squeeze(1)

    return (pos_logits_m, neg_logits_m,
            pos_logits_a, neg_logits_a,
            pos_logits_b, neg_logits_b)