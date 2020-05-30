import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from nltk.tree import Tree

from ..module.nn import CNN
from ..module.pool import MaxPool
from .base_encoder import BaseEncoder
from .library import capsule_fc_layer, CapsTreeLayer, PositionAwareRNN


class CapTreePAEncoder(BaseEncoder):

    def __init__(self, 
                 token2id, num_class, max_length=128, word_size=50, position_size=5, pos_ner=False,
                 blank_padding=True, word2vec=None, activation_function=F.relu,
                 emb_dropout=0.3, rnn_dropout=0.3, cap_dropout=0.5,
                 capsule_size = 16, hidden_capsule_num=16,
                 pretrainLSTM = None, fineTune=True, gamma=0.5
    ):
        hidden_size = hidden_capsule_num * capsule_size

        # Hyperparameters
        super(CapTreePAEncoder, self).__init__(token2id, max_length, hidden_size, word_size, position_size, pos_ner, blank_padding, word2vec)
        self.act = activation_function
        self.hidden_capsule_num = hidden_capsule_num
        self.capsule_size = capsule_size
        self.num_class = num_class
        self.fine_tune = fineTune
        self.gamma = gamma
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


        self.emb_dropout = nn.Dropout(emb_dropout)
        self.rnn_dropout = nn.Dropout(rnn_dropout)

        self.lstm_fc = nn.Linear(self.hidden_size * 3, num_class)  # head_emb, sent_emb, tail_emb

        self.capsule_init = nn.Linear(self.hidden_size, self.hidden_size)
        self.capsule_tree =  CapsTreeLayer( hidden_capsule_num, capsule_size, num_node_type=100,
            nonlinear=F.hardtanh, dropout=0.
        )
        self.capsule_fc = capsule_fc_layer(self.num_class, poses_shape=self.capsule_size, dropout=cap_dropout)
        self.capsule_fcEmb = capsule_fc_layer(hidden_capsule_num, poses_shape=capsule_size, dropout=cap_dropout)

        self.fin_dropout = nn.Dropout(0.5)
        self.final_fc_cat = nn.Linear(self.hidden_size * 2, self.num_class)
        self.final_fc = nn.Linear(self.hidden_size * 3, self.num_class)

        # output_capsule_num, poses_shape, out_poses_shape=None,
        self.capsule_1D = capsule_fc_layer(1, self.capsule_size, out_poses_shape=self.hidden_size * 3, dropout=cap_dropout)

        if pretrainLSTM is not None:
            self.init_LSTM(pretrainLSTM, dontTrainAgain=not fineTune)

    def init_LSTM(self, pretrainLSTM, dontTrainAgain=True):
        states = torch.load(pretrainLSTM)['state_dict']
        self.word_embedding.load_state_dict({"weight": states["sentence_encoder.word_embedding.weight"]})
        self.pos1_embedding.load_state_dict({"weight": states["sentence_encoder.pos1_embedding.weight"]})
        self.pos2_embedding.load_state_dict({"weight": states["sentence_encoder.pos2_embedding.weight"]})
        self.pos_embedding.load_state_dict({"weight": states["sentence_encoder.pos_embedding.weight"]})
        self.ner_embedding.load_state_dict({"weight": states["sentence_encoder.ner_embedding.weight"]})

        lstm_state = {
            "emb.weight": states["sentence_encoder.lstm.emb.weight"],
            "pos_emb.weight": states["sentence_encoder.lstm.pos_emb.weight"],
            "ner_emb.weight": states["sentence_encoder.lstm.ner_emb.weight"],
            "rnn.weight_ih_l0": states["sentence_encoder.lstm.rnn.weight_ih_l0"],
            "rnn.weight_hh_l0": states["sentence_encoder.lstm.rnn.weight_hh_l0"],
            "rnn.bias_ih_l0": states["sentence_encoder.lstm.rnn.bias_ih_l0"],
            "rnn.bias_hh_l0": states["sentence_encoder.lstm.rnn.bias_hh_l0"],
            "attn_layer.ulinear.weight": states["sentence_encoder.lstm.attn_layer.ulinear.weight"],
            "attn_layer.ulinear.bias": states["sentence_encoder.lstm.attn_layer.ulinear.bias"],
            "attn_layer.vlinear.weight": states["sentence_encoder.lstm.attn_layer.vlinear.weight"],
            "attn_layer.wlinear.weight": states["sentence_encoder.lstm.attn_layer.wlinear.weight"],
            "attn_layer.tlinear.weight": states["sentence_encoder.lstm.attn_layer.tlinear.weight"],
            "attn_layer.tlinear.bias": states["sentence_encoder.lstm.attn_layer.tlinear.bias"],
            "pe_emb.weight": states["sentence_encoder.lstm.pe_emb.weight"],
        }
        self.lstm.load_state_dict(lstm_state)
        self.lstm_fc.load_state_dict({"weight": states['fc.weight'], "bias": states['fc.bias']})

        if dontTrainAgain:
            self.lstm.emb.weight.requires_grad_(False)
            self.lstm.pos_emb.weight.requires_grad_(False)
            self.lstm.ner_emb.weight.requires_grad_(False)
            self.lstm.rnn.weight_ih_l0.requires_grad_(False)
            self.lstm.rnn.weight_hh_l0.requires_grad_(False)
            self.lstm.rnn.bias_ih_l0.requires_grad_(False)
            self.lstm.rnn.bias_hh_l0.requires_grad_(False)
            self.lstm.attn_layer.ulinear.weight.requires_grad_(False)
            self.lstm.attn_layer.ulinear.bias.requires_grad_(False)
            self.lstm.attn_layer.vlinear.weight.requires_grad_(False)
            self.lstm.attn_layer.wlinear.weight.requires_grad_(False)
            self.lstm.attn_layer.tlinear.weight.requires_grad_(False)
            self.lstm.attn_layer.tlinear.bias.requires_grad_(False)
            self.lstm.pe_emb.weight.requires_grad_(False)


    def each_tree(self, X, tree):
        if isinstance(tree, Tree):
            X = self.capsule_tree(X, tree)
        else:
            raise TypeError("tree cannot be recognized.")
        return X

    def process_inputTree(self, X, tree, pos):
        e1p, e2p = torch.min(pos).item(), torch.max(pos).item()
        try:
            parent = tree[tree.treeposition_spanning_leaves(e1p, e2p+1)]  # 找最小公共祖先
        except:
            print("aaaaa")
            parent = tree[tree.treeposition_spanning_leaves(e1p, e2p + 1)]  # 找最小公共祖先
        sub = '\t'.join(tree.leaves()).rfind('\t'.join(parent.leaves()))
        bias = len('\t'.join(tree.leaves())[:sub].split('\t')) - 1
        ans_tree = parent

        return X, ans_tree, bias

    def dropPadding(self, rnn_outputs, seq_lengths, trees, pos):
        capsule_output = []
        for i in range(rnn_outputs.shape[0]):
            length = seq_lengths[i]
            # capsInit
            capsule_input = rnn_outputs[i:i+1, :length].view(1, length, self.hidden_capsule_num, self.capsule_size)
            # capsuleTree
            tree_input, tree, bias = self.process_inputTree(capsule_input, trees[i], pos[i])
            tree_input = tree_input[:, bias:, :, :]

            tree_output = self.each_tree(tree_input, tree).view(1, -1, self.capsule_size)
            # capsule_output.append(self.capsule_fc(tree_output))
            # capsule_output.append(self.capsule_fcEmb(tree_output))

            # capsule_output.append(self.capsule_1D(tree_output))
            ent_vec = torch.add(capsule_input[:, pos[i][0]], capsule_input[:, pos[i][1]])
            capsule_output.append(self.capsule_1D(tree_output, ent_vec=ent_vec))

        capsule_output = torch.cat(capsule_output, dim=0)
        return capsule_output

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
        try:
            trees = [Tree.fromstring(t) for t in tree_string]
        except:
            print("tree_string is wrong")
            print(tree_string)
            trees = [Tree.fromstring(t) for t in tree_string]

        if len(token.size()) != 2 or token.size() != pos1.size() or token.size() != pos2.size():
            raise Exception("Size of token, pos1 ans pos2 should be (B, L)")
        idx_no_pad = token.ne(self.token2id['[PAD]'])
        seq_lengths = idx_no_pad.sum(dim=1)
        ent_pos = torch.stack([
            torch.argmax( pos1.eq(self.max_length).int(), dim=1 ),
            torch.argmax( pos2.eq(self.max_length).int(), dim=1 )], dim=1)  # （B, 2)

        inputs = [token, token.eq(self.token2id['[PAD]']), pos, ner, tree_string, pos1, pos2]
        sent_emb, rnn_outputs = self.lstm(inputs)  # (B, hiddenSize) (B, L, E)


        entity_embedding = torch.stack(
            [rnn_outputs[i].index_select(dim=0, index=ent_pos[i]) for i in range(rnn_outputs.shape[0])]
            , dim=0)
        head_emb, tail_emb = entity_embedding[:, 0, :], entity_embedding[:, 1, :]
        logit_lstm = self.lstm_fc(torch.cat([head_emb, sent_emb, tail_emb], dim=1))

        capsule_input = self.capsule_init(rnn_outputs)
        capsule_output = self.dropPadding(capsule_input, seq_lengths, trees, ent_pos)
        capsule_output = capsule_output.view(sent_emb.shape[0], -1)
        logit_capsule = self.final_fc(capsule_output)

        # self.gamma =0
        return logit_capsule * self.gamma + logit_lstm * (1 - self.gamma)

    def tokenize(self, item):
        return super().tokenize(item)




