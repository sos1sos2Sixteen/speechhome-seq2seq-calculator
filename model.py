import torch
import torch.nn as nn 
import random 


class Encoder(nn.Module): 
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout): 
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embeddings = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lengths): 
        # src: (Ts, bcsz)
        # src_lengths: (bcsz,)

        # embedded: (Ts, bcsz, f)
        embedded = self.dropout(self.embeddings(src))

        # outputs: (Ts, bcsz, f)
        # hidden: (nlayers, bcsz, f)
        # cell: (nlayers, bcsz, f)
        # assert outputs[-1] == hidden[-1]
        outputs, (hidden, cell) = self.rnn(embedded)

        # return (hidden, cell)

        # (bcsz, f)
        higher_hidden_accurate = torch.stack([
            outputs[t, i_batch] for i_batch, t in enumerate(src_lengths - 1)
        ], dim=0)
        hidden_out = torch.zeros_like(hidden)
        cell_out = torch.zeros_like(cell)

        hidden_out[0] = higher_hidden_accurate
        return (hidden_out, cell_out)



class Decoder(nn.Module): 
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout): 
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embeddings = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell): 
        # input: (bcsz, )
        # hidden: (nlayers, bcsz, f)
        # cell: (nlayers, bcsz, f)

        # (1, bcsz)
        input = input.unsqueeze(0)
        
        # (1, bcsz, f)
        embedded = self.dropout(self.embeddings(input))

        # output: (1, bcsz, f)
        # hidden: (nlayers, bcsz, f)
        # cell: (nlayers, bcsz, f)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # predictions: (bcsz, output_dim)
        prediction = self.fc_out(output.squeeze(0))

        return prediction, hidden, cell

class Seq2Seq(nn.Module): 

    def __init__(self, encoder, decoder, sos_value, eos_value): 
        super().__init__()
        # for duck-typing 
        self.encoder = encoder
        self.decoder = decoder

        self.sos_value = sos_value
        self.eos_value = eos_value

        assert encoder.hid_dim == decoder.hid_dim
        assert encoder.n_layers == decoder.n_layers
    
    def forward(self, src, src_lengths, trg): 
        # src: (Ts, bcsz)
        # src_lengths: (bcsz, )
        # trg: (Tt, bcsz)

        _, bcsz = src.shape
        Tt, _   = trg.shape
        target_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(Tt, bcsz, target_vocab_size).to(trg.device)

        hidden, cell = self.encoder(src, src_lengths)

        input = trg[0, :]
        for t in range(1, Tt): 
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            input = trg[t]

        return outputs

    def decode(self, src, src_lengths, max_decode_step=20): 
        # src: (Ts, bcsz)
        # src_lengths: (bcsz, )

        _, bcsz = src.shape

        hidden, cell = self.encoder(src, src_lengths)

        input = torch.ones((bcsz, ), dtype=torch.long).to(src.device) * self.sos_value
        eos_count = torch.zeros((bcsz, )).to(src.device)
        
        history = [] 

        for _ in range(max_decode_step): 
            # output: (bcsz, C)
            output, hidden, cell = self.decoder(input, hidden, cell)

            # top1: (bcsz, )
            top1 = output.argmax(1) # greedy search

            eos_count += (top1 == self.eos_value)
            if (eos_count != 0).all(): break

            history.append(top1)
            input = top1.long()
        
        if len(history) == 0: 
            return torch.ones((bcsz, 1), dtype=torch.long).to(src.device) * self.eos_value

        # (bcsz, T)
        return torch.stack(history, dim=-1)




        


