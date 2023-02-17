import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import numpy as np

class Encoder(nn.Module):
    def __init__(self,input_size, hidden_size, dropout_embedding, dropout_encoder):
        super(Encoder, self).__init__()
        self.wordEmbeddingDim = input_size
        self.hidden_size = hidden_size
        self.encoder = nn.GRU(input_size,self.hidden_size, bidirectional=True)
        self.dropout_embedding = nn.Dropout(p = dropout_embedding)
        self.dropout_encoder = nn.Dropout(p = dropout_encoder)

    def forward(self, seq, lens):
        batch_size = seq.shape[0]
        lens_sorted, lens_argsort = torch.sort(lens, 0, True)
        _, lens_argsort_argsort = torch.sort(lens_argsort, 0)
        seq_ = torch.index_select(seq, 0, lens_argsort)
        seq_embd = self.dropout_embedding(seq_)
        packed = pack_padded_sequence(seq_embd, lens_sorted, batch_first=True)

        self.encoder.flatten_parameters()

        output, h = self.encoder(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = output.contiguous()
        output = torch.index_select(output, 0, lens_argsort_argsort)  # B x m x 2l
        #last hidden state
        h = h.permute(1,0,2).contiguous().view(batch_size,1,-1)
        h = torch.index_select(h, 0, lens_argsort_argsort)
        output = self.dropout_encoder(output)
        h = self.dropout_encoder(h)
        return output, h


class LockedDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        dropout = self.dropout
        if not self.training:
            return x
        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x


class EncoderEntity(nn.Module):
    def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last):
        super().__init__()
        self.rnns = []
        for i in range(nlayers):
            if i == 0:
                input_size_ = input_size
                output_size_ = num_units
            else:
                input_size_ = num_units if not bidir else num_units * 2
                output_size_ = num_units
            self.rnns.append(nn.LSTM(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
        self.rnns = nn.ModuleList(self.rnns)

        self.init_hidden = nn.ParameterList(
            [nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
        self.init_c = nn.ParameterList(
            [nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])

        self.dropout = LockedDropout(dropout)
        self.concat = concat
        self.nlayers = nlayers
        self.return_last = return_last

        # self.reset_parameters()

    def reset_parameters(self):
        for rnn in self.rnns:
            for name, p in rnn.named_parameters():
                if 'weight' in name:
                    p.data.normal_(std=0.1)
                else:
                    p.data.zero_()

    def get_init(self, bsz, i):
        return self.init_hidden[i].expand(-1, bsz, -1).contiguous(), self.init_c[i].expand(-1, bsz, -1).contiguous()

    def forward(self, input, input_lengths=None):
        lengths = torch.tensor(input_lengths)
        lens, indices = torch.sort(lengths, 0, True)
        input = input[indices]
        _, _indices = torch.sort(indices, 0)

        bsz, slen = input.size(0), input.size(1)
        output = input
        outputs = []
        # if input_lengths is not None:
        #    lens = input_lengths.data.cpu().numpy()
        lens[lens == 0] = 1

        for i in range(self.nlayers):
            hidden, c = self.get_init(bsz, i)

            output = self.dropout(output)
            if input_lengths is not None:
                output = pack_padded_sequence(output, lens, batch_first=True)

            output, (hidden, c) = self.rnns[i](output, (hidden, c))

            if input_lengths is not None:
                output, _ = pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen:  # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen - output.size(1), output.size(2))],
                                       dim=1)
            if self.return_last:
                outputs.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
            else:
                outputs.append(output)
        for i, output in enumerate(outputs):
            outputs[i] = output[_indices]
        # if self.concat:
        #     return torch.cat(outputs, dim=-1)
        return outputs