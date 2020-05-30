import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from nltk.tree import Tree

from ..module.nn import CNN
from ..module.pool import MaxPool
from .base_encoder import BaseEncoder
from .library import capsule_fc_layer, CapsTreeLayer, attention_blstm


class CapTreeCatEncoder(BaseEncoder):

    def __init__(self, 
                 token2id, num_class, max_length=128, word_size=50, position_size=5, pos_ner=False,
                 blank_padding=True, word2vec=None, activation_function=F.relu,
                 emb_dropout=0.3, rnn_dropout=0.3, cap_dropout=0.5,
                 capsule_size = 16, hidden_capsule_num=16,
                 pretrainLSTM = None, fineTune=True, gamma=0.5
    ):
        hidden_size = hidden_capsule_num * capsule_size
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
        super(CapTreeCatEncoder, self).__init__(token2id, max_length, hidden_size, word_size, position_size, pos_ner, blank_padding, word2vec)
        self.act = activation_function
        self.hidden_capsule_num = hidden_capsule_num
        self.capsule_size = capsule_size
        self.num_class = num_class
        self.fine_tune = fineTune
        self.gamma = gamma

        initializer = nn.init.xavier_normal_

        self.emb_dropout = nn.Dropout(emb_dropout)
        self.rnn_dropout = nn.Dropout(rnn_dropout)
        self.cell_fw = nn.LSTMCell(input_size=self.input_size, hidden_size=self.hidden_size)
        self.cell_bw = nn.LSTMCell(input_size=self.input_size, hidden_size=self.hidden_size)
        self.attention = attention_blstm(self.hidden_size)
        self.lstm_fc = nn.Linear(self.hidden_size * 3, num_class)  # head_emb, sent_emb, tail_emb

        self.capsule_init = nn.Linear(self.hidden_size, self.hidden_size)
        self.capsule_tree =  CapsTreeLayer( hidden_capsule_num, capsule_size, num_node_type=100,
            nonlinear=F.hardtanh, dropout=0.5
        )
        self.capsule_fc = capsule_fc_layer(self.num_class, poses_shape=self.capsule_size, dropout=cap_dropout)
        self.capsule_fcEmb = capsule_fc_layer(hidden_capsule_num, poses_shape=capsule_size, dropout=cap_dropout)

        self.fin_dropout = nn.Dropout(0.5)
        self.final_fc_cat = nn.Linear(self.hidden_size * 2, self.num_class)
        self.final_fc = nn.Linear(self.hidden_size * 3, self.num_class)

        # output_capsule_num, poses_shape, out_poses_shape=None,
        # self.capsule_1D = capsule_fc_layer(1, self.capsule_size, out_poses_shape=self.hidden_size * 3, dropout=cap_dropout)
        self.capsule_1D = capsule_fc_layer(1, self.capsule_size, out_poses_shape=self.hidden_size * 3,
                                           nonlinear=F.hardtanh, dropout=cap_dropout)

        if pretrainLSTM is not None:
            self.init_LSTM(pretrainLSTM, dontTrainAgain=not fineTune)
        self.group_para()

    def group_para(self):
        # Group 1
        self.cell_fw.state_dict()['weight_hh'].group = 1
        self.cell_fw.state_dict()['weight_ih'].group = 1
        self.cell_fw.state_dict()['bias_hh'].group = 1
        self.cell_fw.state_dict()['bias_ih'].group = 1
        self.cell_bw.state_dict()['weight_hh'].group = 1
        self.cell_bw.state_dict()['weight_ih'].group = 1
        self.cell_bw.state_dict()['bias_hh'].group = 1
        self.cell_bw.state_dict()['bias_ih'].group = 1
        self.attention.state_dict()['u_omega'].group = 1
        self.lstm_fc.state_dict()['weight'].group = 1
        self.lstm_fc.state_dict()['bias'].group = 1
        # Group 2
        self.capsule_fc.vec_transformation.kernel.group = 2
        # Group 3
        for fc in self.capsule_tree.fc:
            fc.vec_transformation.kernel.group = 3

    def init_LSTM(self, pretrainLSTM, dontTrainAgain=True):
        states = torch.load(pretrainLSTM)['state_dict']
        fw_state = {
            'weight_ih': states['sentence_encoder.cell_fw.weight_ih'],
            'weight_hh': states['sentence_encoder.cell_fw.weight_hh'],
            'bias_ih': states['sentence_encoder.cell_fw.bias_ih'],
            'bias_hh': states['sentence_encoder.cell_fw.bias_hh'],
        }
        bw_state = {
            'weight_ih': states['sentence_encoder.cell_bw.weight_ih'],
            'weight_hh': states['sentence_encoder.cell_bw.weight_hh'],
            'bias_ih': states['sentence_encoder.cell_bw.bias_ih'],
            'bias_hh': states['sentence_encoder.cell_bw.bias_hh'],
        }
        self.cell_fw.load_state_dict(fw_state)
        self.cell_bw.load_state_dict(bw_state)
        self.word_embedding.load_state_dict({'weight': states['sentence_encoder.word_embedding.weight']})
        self.pos1_embedding.load_state_dict({'weight': states['sentence_encoder.pos1_embedding.weight']})
        self.pos2_embedding.load_state_dict({'weight': states['sentence_encoder.pos2_embedding.weight']})
        self.attention.load_state_dict({"u_omega": states["sentence_encoder.attention.u_omega"]})
        if 'fc.weight' in states:
            self.lstm_fc.load_state_dict({"weight": states['fc.weight'], "bias": states['fc.bias']})
        else:
            self.lstm_fc.load_state_dict({"weight": states['sentence_encoder.lstm_fc.weight'],
                                          "bias": states['sentence_encoder.lstm_fc.bias']})
            self.capsule_init.load_state_dict({"weight": states['sentence_encoder.capsule_init.weight'],
                                               "bias": states['sentence_encoder.capsule_init.bias']})
            self.capsule_fc.load_state_dict({
                'vec_transformation.kernel': states['sentence_encoder.capsule_fc.vec_transformation.kernel'],
                'beta_a': states['sentence_encoder.capsule_fc.beta_a']})
            if dontTrainAgain:
                self.capsule_init.weight.requires_grad_(False)
                self.capsule_init.bias.requires_grad_(False)
                self.capsule_fc.vec_transformation.kernel.requires_grad_(False)
        if dontTrainAgain:
            self.cell_fw.weight_hh.requires_grad_(False)
            self.cell_fw.weight_ih.requires_grad_(False)
            self.cell_fw.bias_hh.requires_grad_(False)
            self.cell_fw.bias_ih.requires_grad_(False)
            self.cell_bw.weight_hh.requires_grad_(False)
            self.cell_bw.weight_ih.requires_grad_(False)
            self.cell_bw.bias_hh.requires_grad_(False)
            self.cell_bw.bias_ih.requires_grad_(False)
            self.attention.u_omega.requires_grad_(False)
            self.lstm_fc.weight.requires_grad_(False)
            self.lstm_fc.bias.requires_grad_(False)

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
        return output

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

    def get_embedding(self, token, pos1, pos2, pos, ner):
        if len(token.size()) != 2 or token.size() != pos1.size() or token.size() != pos2.size():
            raise Exception("Size of token, pos1 ans pos2 should be (B, L)")
        idx_no_pad = token.ne(self.token2id['[PAD]'])
        seq_lengths = idx_no_pad.sum(dim=1)
        ent_pos = torch.stack([
            torch.argmax( pos1.eq(self.max_length).int(), dim=1 ),
            torch.argmax( pos2.eq(self.max_length).int(), dim=1 )], dim=1)  # （B, 2)

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
        return embedded_chars

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

        embedded_chars = self.get_embedding(token, pos1, pos2, pos, ner)
        # Dropout for Word Embedding
        embedded_chars = self.emb_dropout(embedded_chars)
        # LSTM
        rnn_outputs = self.LSTM(embedded_chars)  # (B, L, EMBED)
        attn, alphas = self.attention(rnn_outputs)
        entity_embedding = torch.stack(
            [rnn_outputs[i].index_select(dim=0, index=ent_pos[i]) for i in range(rnn_outputs.shape[0])]
            , dim=0)
        head_emb, tail_emb = entity_embedding[:, 0, :], entity_embedding[:, 1, :]
        sent_emb = attn
        logit_lstm = self.lstm_fc(torch.cat([head_emb, sent_emb, tail_emb], dim=1))

        # embedded_chars, seq_lengths, trees, pos
        capsule_input = self.capsule_init(rnn_outputs)
        # capsule_input = rnn_outputs
        capsule_output = self.dropPadding(capsule_input, seq_lengths, trees, ent_pos)
        # activations
        # logit_capsule = torch.sqrt(torch.sum(capsule_output ** 2, dim=-1))

        # return logit_capsule * self.gamma + logit_lstm * (1 - self.gamma)
        # pred_capusle, pred_lstm = logit_capsule, F.softmax(logit_lstm, dim=1)

        capsule_output = capsule_output.view(sent_emb.shape[0], -1)
        logit_capsule = self.final_fc(capsule_output)


        # CASE_STUDY
        self.logit_capsule = logit_capsule
        self.logit_lstm = logit_lstm


        # self.gamma =0
        return logit_capsule * self.gamma + logit_lstm * (1 - self.gamma)
        # return logit_capsule #* self.gamma + logit_lstm * (1 - self.gamma)


        # capsule_output = capsule_output.view(-1, self.hidden_size)
        # outputs = torch.cat([attn, capsule_output], dim=1)
        # outputs = self.fin_dropout(outputs)
        # outputs = self.final_fc_cat(outputs)
        # return outputs

    def tokenize(self, item):
        return super().tokenize(item)




