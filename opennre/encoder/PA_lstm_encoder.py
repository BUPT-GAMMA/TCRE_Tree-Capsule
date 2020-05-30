import torch
import torch.nn as nn
import torch.nn.functional as F

from ..module.nn import CNN
from ..module.pool import MaxPool
from .base_encoder import BaseEncoder
from .library import PositionAwareAttention, PositionAwareRNN
DEBUG = True

class PALSTMEncoder(BaseEncoder):

    def __init__(self, 
                 token2id, max_length=128, hidden_size=230, word_size=50, position_size=5, pos_ner=False,
                 blank_padding=True, word2vec=None, activation_function=F.relu,
                 emb_dropout=0.3, rnn_dropout=0.3#, dropout=0.5,
                 ):
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
        super(PALSTMEncoder, self).__init__(token2id, max_length, hidden_size, word_size, position_size, pos_ner, blank_padding, word2vec)
        opt = {
            'dropout': rnn_dropout,         'max_length': max_length,
            'vocab_size': self.num_token-2, 'emb_dim': self.word_size,
            'pos_dim': position_size,       'ner_dim': position_size,
            'hidden_dim': hidden_size,      'num_layers': 1,
            'topn': 1e10,                   'pe_dim': position_size,
            'attn_dim': 200,                'cuda': torch.cuda.is_available(),
            'attn': True,
        }
        self.lstm = PositionAwareRNN(opt, self.word2vec)

        # self.hidden_size *= 3

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
        idx_no_pad = token.ne(self.token2id['[PAD]'])
        seq_lengths = idx_no_pad.sum(dim=1).long()
        ent_pos = torch.stack([      # (batch, 2)
            torch.argmax(pos1.eq(self.max_length).int(), dim=1),
            torch.argmax(pos2.eq(self.max_length).int(), dim=1)], dim=1)

        inputs = [token, token.eq(self.token2id['[PAD]']), pos, ner, tree_string, pos1, pos2]
        # Bidirectional LSTM
        sent_emb, rnn_outputs = self.lstm(inputs)  # (B, hiddenSize) (B, L, E)

        entity_embedding = torch.stack(
            [rnn_outputs[i].index_select(dim=0, index=ent_pos[i]) for i in range(rnn_outputs.shape[0])]
            , dim=0)
        head_emb, tail_emb = entity_embedding[:, 0, :], entity_embedding[:, 1, :]

        outputs = torch.cat([head_emb, sent_emb, tail_emb], dim=1)
        # Dropout
        # 最后一个Dropout在softmax_nn.py里，这里就不需要了
        return sent_emb
        # return outputs

    def tokenize(self, item):
        return super().tokenize(item)



