import torch
import torch.nn as nn
import torch.nn.functional as F

from ..module.nn import CNN
from ..module.pool import MaxPool
from .base_encoder import BaseEncoder
from .library import attention_blstm

class LSTMEntityEncoder(BaseEncoder):

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
        super(LSTMEntityEncoder, self).__init__(token2id, max_length, hidden_size, word_size, position_size, pos_ner, blank_padding, word2vec)
        self.act = activation_function

        initializer = nn.init.xavier_normal_

        self.emb_dropout = nn.Dropout(emb_dropout)
        self.rnn_dropout = nn.Dropout(rnn_dropout)

        LSTM_outputSize = self.hidden_size #// 2
        self.cell_fw = nn.LSTMCell(input_size=self.input_size, hidden_size=LSTM_outputSize)
        self.cell_bw = nn.LSTMCell(input_size=self.input_size, hidden_size=LSTM_outputSize)
        self.attention = attention_blstm(self.hidden_size)

        self.hidden_size *= 3   # head-sent-tail


    def LSTM(self, inputs):
        # (B, L, EMBED) -> (B, L, EMBED)
        x = inputs.transpose(0, 1)  # (L, B, EMBED)
        output_fw, output_bw = [], []
        for i in range(x.shape[0]):
            # (batch, feature)
            inp_fw = self.rnn_dropout(x[i])
            inp_bw = self.rnn_dropout(x[x.shape[0] - i - 1])
            if i == 0:
                hx_fw, cx_fw = self.cell_fw(inp_fw)
                hx_bw, cx_bw = self.cell_bw(inp_bw)
            else:
                hx_fw, cx_fw = self.cell_fw(inp_fw, (hx_fw, cx_fw))
                hx_bw, cx_bw = self.cell_bw(inp_bw, (hx_bw, cx_bw))
            output_fw.append(hx_fw)
            output_bw.append(hx_bw)

        output_fw = torch.stack(output_fw, dim=1)
        output_bw = torch.stack(output_bw[::-1], dim=1)
        output = torch.add(output_fw, output_bw)
        # output = torch.cat([output_fw, output_bw], dim=-1)
        return output

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
        seq_lengths = idx_no_pad.sum(dim=1)
        ent_pos = torch.stack([      # (batch, 2)
            torch.argmax(pos1.eq(self.max_length).int(), dim=1),
            torch.argmax(pos2.eq(self.max_length).int(), dim=1)], dim=1)
        if pos is None:
            embedded_chars = torch.cat([self.word_embedding(token),
                           self.pos1_embedding(pos1),
                           self.pos2_embedding(pos2)], 2) # (B, L, EMBED)
        else:
            embedded_chars = torch.cat([self.word_embedding(token),
                                        self.pos1_embedding(pos1),
                                        self.pos2_embedding(pos2),
                                        self.pos_embedding(pos),
                                        self.ner_embedding(ner)], 2)  # (B, L, EMBED)
        # Dropout for Word Embedding
        embedded_chars = self.emb_dropout(embedded_chars)

        drop_the_paddings = False
        if drop_the_paddings:
            attn = []
            for i in range(embedded_chars.shape[0]):
                # Bidirectional LSTM
                length = seq_lengths[i]
                rnn_outputs = self.LSTM(embedded_chars[i:i + 1, :length])  # (B, L, EMBED)
                # Attention
                out, alphas = self.attention(rnn_outputs)
                attn.append(out)
            attn = torch.cat(attn, dim=0)

        else:
            # Bidirectional LSTM
            rnn_outputs = self.LSTM(embedded_chars)
            # Attention
            attn, alphas = self.attention(rnn_outputs)    # # (batch, hiddenSize), (batch, maxLength)

        sent_emb = attn   # (batch, hiddenSize)

        entity_embedding = torch.stack(
            [rnn_outputs[i].index_select(dim=0, index=ent_pos[i]) for i in range(rnn_outputs.shape[0])]
            , dim=0)
        head_emb, tail_emb = entity_embedding[:, 0, :], entity_embedding[:, 1, :]

        outputs = torch.cat([head_emb, sent_emb, tail_emb], dim=1)
        # Dropout
        # 最后一个Dropout在softmax_nn.py里，这里就不需要了
        return outputs

    def tokenize(self, item):
        return super().tokenize(item)



