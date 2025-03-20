import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.Embedding = nn.Linear(6, cfg.embed, bias=False)
        self.Encoder = nn.LSTM(input_size=cfg.embed, hidden_size=cfg.hidden, batch_first=True)
        self.Decoder = nn.LSTM(input_size=cfg.embed, hidden_size=cfg.hidden, batch_first=True)
        self.Vec = nn.Parameter(torch.FloatTensor(cfg.embed))
        self.W_q = nn.Linear(cfg.hidden, cfg.hidden, bias=True)
        self.W_ref = nn.Conv1d(cfg.hidden, cfg.hidden, 1, 1)
        self.final2FC = nn.Sequential(
            nn.Linear(cfg.hidden, cfg.hidden, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(cfg.hidden, 1, bias=False))
        self._initialize_weights(cfg.init_min, cfg.init_max)
        self.n_glimpse = cfg.n_glimpse
        self.n_process = cfg.n_process

    def _initialize_weights(self, init_min=-0.08, init_max=0.08):
        for param in self.parameters():
            nn.init.uniform_(param.data, init_min, init_max)

    def forward(self, raw_inputs):
        """
        :arg
            x: (batch, task_n, 6)
            enc_h: (batch, task_n, embed)
            query(Decoder input): (batch, 1, embed)
            h: (1, batch, embed)
        :return
            pred_l: (batch)
        """
        embed_enc_inputs = self.Embedding(raw_inputs)
        enc_h, (h, c) = self.Encoder(embed_enc_inputs, None)
        ref = enc_h
        query = h[-1]
        for _ in range(self.n_process):
            for _ in range(self.n_glimpse):
                query = self.glimpse(query, ref)

        pred_l = self.final2FC(query).squeeze(-1).squeeze(-1)
        return pred_l

    def glimpse(self, query, ref):
        """
        :arg
            query: the hidden state of the decoder at the current
            (batch, 128)
            ref: the set of hidden states from the encoder.
            (batch, task_n, 128)
        """
        u1 = self.W_q(query).unsqueeze(-1).repeat(1, 1, ref.size(1))  # u1: (batch, 128, city_t)
        u2 = self.W_ref(ref.permute(0, 2, 1))  # u2: (batch, 128, city_t)
        V = self.Vec.unsqueeze(0).unsqueeze(0).repeat(ref.size(0), 1, 1)
        u = torch.bmm(V, torch.tanh(u1 + u2)).squeeze(1)
        # V: (batch, 1, 128) * u1+u2: (batch, 128, city_t) => u: (batch, 1, city_t) => (batch, city_t)
        a = F.softmax(u, dim=1)
        d = torch.bmm(u2, a.unsqueeze(2)).squeeze(2)
        # u2: (batch, 128, city_t) * a: (batch, city_t, 1) => d: (batch, 128)
        return d
