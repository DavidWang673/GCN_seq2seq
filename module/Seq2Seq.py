import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    """docstring for Encoder"""

    def __init__(self, encoder_size):

        super(Encoder, self).__init__()

        self.input_size,\
            self.hidden_size = encoder_size

        self.encoder = nn.LSTM(
            self.input_size, self.hidden_size, batch_first=True)

    def forward(self, x):
        out, hidden = self.encoder(x)
        return out, hidden


class Decoder(nn.Module):
    """docstring for Decoder"""

    def __init__(self, decoder_size, teacher_forcing=False):
        super(Decoder, self).__init__()

        self.input_size, self.hidden_size = decoder_size

        self.teacher_forcing = teacher_forcing

        # self.encoder_hidden = encoder_hidden

        self.decoder = nn.LSTM(
            self.input_size, self.hidden_size, batch_first=True)

        self.W = nn.Parameter(
            tc.zeros(size=(self.hidden_size, self.input_size)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

    def forward(self, target, hidden):

        out, hidden = self.decoder(target, hidden)

        return tc.matmul(out, self.W), hidden


# class Seq2Seq(nn.Module):
# 	"""docstring for Seq2Seq"""
# 	def __init__(self, encoder_size, decoder_size):
# 		super(Seq2Seq, self).__init__()


# 		self.encoder_rnn = Encoder(encoder_size)
# 		self.decoder_rnn = Decoder(decoder_size)

# 	def forward(self, ):
