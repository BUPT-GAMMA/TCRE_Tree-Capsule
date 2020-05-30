import torch
import torch.nn as nn
import torch.nn.functional as F

from ..module.nn import CNN
from ..module.pool import MaxPool
from .base_encoder import BaseEncoder

class CNNEncoder(BaseEncoder):

    def __init__(self, 
                 token2id, 
                 max_length=128, 
                 hidden_size=230, 
                 word_size=50,
                 position_size=5,
                 pos_ner=False,
                 blank_padding=True,
                 word2vec=None,
                 kernel_size=3, 
                 padding_size=1,
                 dropout=0,
                 activation_function=F.relu):
        """
        Args:
            token2id: dictionary of token->idx mapping
            max_length: max length of sentence, used for postion embedding
            hidden_size: hidden size
            word_size: size of word embedding
            position_size: size of position embedding
            blank_padding: padding for CNN
            word2vec: pretrained word2vec numpy
            kernel_size: kernel_size size for CNN
            padding_size: padding_size for CNN
        """
        # Hyperparameters
        super(CNNEncoder, self).__init__(token2id, max_length, hidden_size, word_size, position_size, pos_ner, blank_padding, word2vec)
        self.drop = nn.Dropout(dropout)
        self.kernel_size = kernel_size
        self.padding_size = padding_size
        self.act = activation_function
        if isinstance(self.kernel_size, int):   self.kernel_size = list(self.kernel_size)
        if isinstance(self.padding_size, int):   self.padding_size = list(self.padding_size)

        self.convs = nn.ModuleList()
        for i in range(len(self.kernel_size)):
            self.convs.append(nn.Conv1d(self.input_size, self.hidden_size,
                                       self.kernel_size[i], padding=self.padding_size[i]))
        self.pool = nn.MaxPool1d(self.max_length)
        self.hidden_size *= len(self.kernel_size)

    def forward(self, token, pos1, pos2, tree_string=None, pos=None, ner=None):
        """
        Args:
            token: (B, L), index of tokens
            pos1: (B, L), relative position to head entity
            pos2: (B, L), relative position to tail entity
        Return:
            (B, EMBED), representations for sentences
        """
        # Check size of tensors
        if len(token.size()) != 2 or token.size() != pos1.size() or token.size() != pos2.size():
            raise Exception("Size of token, pos1 ans pos2 should be (B, L)")
        if pos is None:
            x = torch.cat([self.word_embedding(token),
                           self.pos1_embedding(pos1),
                           self.pos2_embedding(pos2)], 2) # (B, L, EMBED)
        else:
            x = torch.cat([self.word_embedding(token),
                            self.pos1_embedding(pos1),
                            self.pos2_embedding(pos2),
                            self.pos_embedding(pos),
                            self.ner_embedding(ner)], 2)  # (B, L, EMBED)
        x = x.transpose(1, 2) # (B, EMBED, L)
        xs = []
        for conv in self.convs:
            cache = self.act(conv(x)) # (B, H, L)
            xs.append(self.pool(cache).squeeze(-1))
        x = torch.cat(xs, dim=1)
        x = self.drop(x)
        return x

    def tokenize(self, item):
        return super().tokenize(item)
