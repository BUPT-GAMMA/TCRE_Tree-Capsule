#!/user/bin/env python
# -*- coding: utf-8 -*-

import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

class TREE(object):
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth


class attention_blstm(nn.Module):
    def __init__(self, hidden_size):
        super(attention_blstm, self).__init__()
        self.hidden_size = hidden_size
        self.u_omega = nn.Parameter(torch.empty((hidden_size, 1)))
        nn.init.xavier_normal_(self.u_omega)

    def forward(self, inputs):
        v = torch.tanh(inputs)
        vu = torch.matmul(v, self.u_omega).squeeze(-1)
        alphas = torch.softmax(vu, dim=1)

        output = torch.matmul(alphas.unsqueeze(dim=1), inputs).squeeze(dim=1)
        output = torch.tanh(output)
        return output, alphas


class PositionAwareAttention(nn.Module):
    """
    A position-augmented attention layer where the attention weight is
    a = T' . tanh(Ux + Vq + Wf)
    where x is the input, q is the query, and f is additional position features.
    """

    def __init__(self, input_size, query_size, feature_size, attn_size):
        super(PositionAwareAttention, self).__init__()
        self.input_size = input_size
        self.query_size = query_size
        self.feature_size = feature_size
        self.attn_size = attn_size
        self.ulinear = nn.Linear(input_size, attn_size)
        self.vlinear = nn.Linear(query_size, attn_size, bias=False)
        if feature_size > 0:
            self.wlinear = nn.Linear(feature_size, attn_size, bias=False)
        else:
            self.wlinear = None
        self.tlinear = nn.Linear(attn_size, 1)
        self.init_weights()

    def init_weights(self):
        self.ulinear.weight.data.normal_(std=0.001)
        self.vlinear.weight.data.normal_(std=0.001)
        if self.wlinear is not None:
            self.wlinear.weight.data.normal_(std=0.001)
        self.tlinear.weight.data.zero_()  # use zero to give uniform attention at the beginning

    def forward(self, x, x_mask, q, f):
        """
        x : batch_size * seq_len * input_size
        q : batch_size * query_size
        f : batch_size * seq_len * feature_size
        """
        batch_size, seq_len, _ = x.size()

        x_proj = self.ulinear(x.contiguous().view(-1, self.input_size)).view(
            batch_size, seq_len, self.attn_size)
        q_proj = self.vlinear(q.view(-1, self.query_size)).contiguous().view(
            batch_size, self.attn_size).unsqueeze(1).expand(
            batch_size, seq_len, self.attn_size)
        if self.wlinear is not None:
            f_proj = self.wlinear(f.view(-1, self.feature_size)).contiguous().view(
                batch_size, seq_len, self.attn_size)
            projs = [x_proj, q_proj, f_proj]
        else:
            projs = [x_proj, q_proj]
        scores = self.tlinear(torch.tanh(sum(projs)).view(-1, self.attn_size)).view(
            batch_size, seq_len)

        # mask padding
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        weights = F.softmax(scores, dim=1)
        # weighted average input vectors
        outputs = weights.unsqueeze(1).bmm(x).squeeze(1)
        return outputs

class PositionAwareRNN(nn.Module):
    """ A sequence model for relation extraction. """

    def __init__(self, opt, emb_matrix=None):
        super(PositionAwareRNN, self).__init__()
        self.PAD_ID = 0
        UNK_TOKEN = '[UNK]'
        PAD_TOKEN = '[PAD]'
        self.NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'O': 2, 'PERSON': 3, 'ORGANIZATION': 4, 'LOCATION': 5, 'DATE': 6, 'NUMBER': 7, 'MISC': 8, 'DURATION': 9, 'MONEY': 10, 'PERCENT': 11, 'ORDINAL': 12, 'TIME': 13, 'SET': 14}
        self.POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'NNP': 2, 'NN': 3, 'IN': 4, 'DT': 5, ',': 6, 'JJ': 7, 'NNS': 8, 'VBD': 9, 'CD': 10, 'CC': 11, '.': 12, 'RB': 13, 'VBN': 14, 'PRP': 15, 'TO': 16, 'VB': 17, 'VBG': 18, 'VBZ': 19, 'PRP$': 20, ':': 21, 'POS': 22, '\'\'': 23, '``': 24, '-RRB-': 25, '-LRB-': 26, 'VBP': 27, 'MD': 28, 'NNPS': 29, 'WP': 30, 'WDT': 31, 'WRB': 32, 'RP': 33, 'JJR': 34, 'JJS': 35, '$': 36, 'FW': 37, 'RBR': 38, 'SYM': 39, 'EX': 40, 'RBS': 41, 'WP$': 42, 'PDT': 43, 'LS': 44, 'UH': 45, '#': 46}
        self.MAX_LEN = opt['max_length']

        self.drop = nn.Dropout(opt['dropout'])
        self.emb = nn.Embedding(opt['vocab_size']+2, opt['emb_dim'], padding_idx=self.PAD_ID)
        if opt['pos_dim'] > 0:
            self.pos_emb = nn.Embedding(len(self.POS_TO_ID), opt['pos_dim'],
                                        padding_idx=self.PAD_ID)
        if opt['ner_dim'] > 0:
            self.ner_emb = nn.Embedding(len(self.NER_TO_ID), opt['ner_dim'],
                                        padding_idx=self.PAD_ID)

        # input_size = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim']
        input_size = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim'] + 2 * opt['pe_dim']
        self.rnn = nn.LSTM(input_size, opt['hidden_dim'], opt['num_layers'],
                           batch_first=True, dropout=opt['dropout'])
        # self.linear = nn.Linear(opt['hidden_dim'], opt['num_class'])

        if opt['attn']:
            self.attn_layer = PositionAwareAttention(opt['hidden_dim'],
                              opt['hidden_dim'], 2 * opt['pe_dim'], opt['attn_dim'])
            self.pe_emb = nn.Embedding(self.MAX_LEN * 2, opt['pe_dim'])

        self.opt = opt
        self.topn = self.opt.get('topn', 1e10)
        self.use_cuda = opt['cuda']
        self.emb_matrix = emb_matrix
        self.init_weights()

    def init_weights(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:, :].uniform_(-1.0, 1.0)  # keep padding dimension to be 0
        else:
            unk = torch.randn(1, self.opt['emb_dim']) / math.sqrt(self.opt['emb_dim'])
            blk = torch.zeros(1, self.opt['emb_dim'])
            self.emb.weight.data.copy_(torch.cat([self.emb_matrix, unk, blk], 0))

        if self.opt['pos_dim'] > 0:
            self.pos_emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        if self.opt['ner_dim'] > 0:
            self.ner_emb.weight.data[1:, :].uniform_(-1.0, 1.0)

        # self.linear.bias.data.fill_(0)
        # nn.init.xavier_uniform_(self.linear.weight, gain=1)  # initialize linear layer
        if self.opt['attn']:
            self.pe_emb.weight.data.uniform_(-1.0, 1.0)

        # decide finetuning
        if self.topn <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        # elif self.topn < self.opt['vocab_size']:
        #     print("Finetune top {} word embeddings.".format(self.topn))
        #     self.emb.weight.register_hook(lambda x: \
        #                                       torch_utils.keep_partial_grad(x, self.topn))
        else:
            print("Finetune all embeddings.")

    # def zero_state(self, batch_size):
    #     state_shape = (self.opt['num_layers'], batch_size, self.opt['hidden_dim'])
    #     h0 = c0 = torch.zeros(*state_shape, requires_grad=False)
    #     if self.use_cuda:
    #         return h0.cuda(), c0.cuda()
    #     else:
    #         return h0, c0

    def forward(self, inputs):
        words, masks, pos, ner, trees, subj_pos, obj_pos = inputs  # unpack
        seq_lens = masks.data.eq(self.PAD_ID).long().sum(1).squeeze()
        batch_size = words.size()[0]

        # embedding lookup
        word_inputs = self.emb(words)
        inputs = [word_inputs]
        if self.opt['pos_dim'] > 0:
            inputs += [self.pos_emb(pos)]
        if self.opt['ner_dim'] > 0:
            inputs += [self.ner_emb(ner)]

        if self.opt['pe_dim'] > 0:
            inputs += [self.pe_emb(subj_pos)]
            inputs += [self.pe_emb(obj_pos)]

        inputs = self.drop(torch.cat(inputs, dim=2))  # add dropout to input
        input_size = inputs.size(2)

        # sorted for pack
        _, idx_sort = torch.sort(seq_lens, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        lens = list(seq_lens[idx_sort])

        inputs = nn.utils.rnn.pack_padded_sequence(inputs[idx_sort], lens, batch_first=True)
        outputs, (ht, ct) = self.rnn(inputs)
        outputs, output_lens = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, total_length=self.MAX_LEN)

        # unsort for unpack
        rnn_outputs = outputs.index_select(0, idx_unsort)
        ht = ht.index_select(1, idx_unsort)
        ct = ct.index_select(1, idx_unsort)
        outputs = rnn_outputs

        hidden = self.drop(ht[-1, :, :])  # get the outmost layer h_n
        outputs = self.drop(outputs)

        # pos 是错的

        # attention
        if self.opt['attn']:
            # convert all negative PE numbers to positive indices
            # e.g., -2 -1 0 1 will be mapped to 98 99 100 101
            subj_pe_inputs = self.pe_emb(subj_pos)
            obj_pe_inputs = self.pe_emb(obj_pos)
            pe_features = torch.cat((subj_pe_inputs, obj_pe_inputs), dim=2)
            final_hidden = self.attn_layer(outputs, masks, hidden, pe_features)
        else:
            final_hidden = hidden

        # logits = self.linear(final_hidden)
        return final_hidden, rnn_outputs


def squash(x, dim=-1):
    s_squared_norm = torch.sum(x ** 2, dim, keepdim=True) + 1e-10
    scale = torch.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x

class capsule_fc_layer(nn.Module):
    def __init__(self, output_capsule_num, poses_shape, out_poses_shape=None,
                 iterations=3, nonlinear=squash, dropout=0.):
        super(capsule_fc_layer, self).__init__()
        self.iterations = iterations
        self.output_capsule_num = output_capsule_num
        if out_poses_shape is None:
            out_poses_shape = poses_shape
        self.in_poses_shape = poses_shape
        self.out_poses_shape = out_poses_shape
        self.nonlinear = nonlinear
        self.vec_transformation = vec_transform(
            poses_shape, #in_num,
            out_poses_shape, output_capsule_num,
        )
        self.beta_a = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(1, output_capsule_num))
        )
        self.dropout = capsule_dropout(dropout)

    def forward(self, poses, idx_no_pad=None, ent_vec=None):
        poses = self.dropout(poses)
        input_pose_shape = poses.shape
        u_hat_vecs = self.vec_transformation(poses)
        if ent_vec is None:
            poses = routing(u_hat_vecs, self.beta_a,
                        self.iterations, self.output_capsule_num, self.nonlinear)
        else:
            ent_vec = ent_vec.sum(dim=1, keepdim=True)
            ent_vec = self.vec_transformation(ent_vec)
            poses = att_routing(u_hat_vecs, ent_vec,
                            self.iterations, self.output_capsule_num, self.nonlinear)
        return poses

class capsule_dropout(nn.Module):
    def __init__(self, dropout):
        super(capsule_dropout, self).__init__()
        self.dropout_prob = dropout
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, inputs):
        return self.dropout(inputs)


def routing(*args):
    return raw_routing(*args)

def raw_routing(u_hat_vecs, beta_a, iterations, output_capsule_num, nonlinear=squash):
    b = torch.zeros_like(u_hat_vecs[:, :, :, 0])
    b = torch.ones_like(u_hat_vecs[:, :, :, 0])
    outputs = None
    for i in range(iterations):
        c = F.softmax(b, dim=1)
        # c = b / (b.sum(dim=1) + 1e-10)
        outputs = nonlinear(torch.matmul(c.unsqueeze(2), u_hat_vecs).squeeze(2))
        if i < iterations - 1:
            b = b + torch.matmul(outputs.unsqueeze(2), u_hat_vecs.transpose(2, 3)).squeeze(2)
    return outputs

def att_routing(u_hat_vecs, entity_features, iterations, output_capsule_num, nonlinear=squash):
    if isinstance(entity_features, list):
        entity_features = entity_features[0] + entity_features[1]

    b = torch.zeros_like(u_hat_vecs[:, :, :, 0])
    b = torch.ones_like(u_hat_vecs[:, :, :, 0])
    outputs = None
    for i in range(iterations):
        w = torch.softmax(b, dim=1)
        # w = b / (b.sum(dim=1) + 1e-10)
        a = torch.sigmoid(torch.matmul(u_hat_vecs, entity_features.transpose(-1, -2))).squeeze(-1)
        w = torch.mul(w, a).unsqueeze(2)
        outputs = nonlinear(torch.matmul(w, u_hat_vecs).squeeze(2))
        if i < iterations - 1:
            b = b + torch.matmul(outputs.unsqueeze(2), u_hat_vecs.transpose(2, 3)).squeeze(2)
    return outputs

def routing_for_dropout(u_hat_vecs, beta_a, iterations, output_capsule_num, nonlinear=squash):
    b = torch.ones_like(u_hat_vecs[:, :, :, 0])    # B, Sent, Num, Dim
    zeros = torch.ones_like(b) * 1e-19
    outputs = None
    for i in range(iterations):
        c = F.softmax(b, dim=1)
        outputs = nonlinear(torch.matmul(c.unsqueeze(2), u_hat_vecs).squeeze(2))
        if i < iterations - 1:
            b = b + torch.matmul(outputs.unsqueeze(2), u_hat_vecs.transpose(2, 3)).squeeze(2)
    return outputs

class vec_transform(nn.Module):
    def __init__(self, input_capsule_dim, #input_capsule_num,
                 output_capsule_dim, output_capsule_num):
        super(vec_transform, self).__init__()
        self.kernel = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(
                (output_capsule_dim*output_capsule_num, input_capsule_dim, 1)))
        )
        self.input_capsule_dim = input_capsule_dim
        self.output_capsule_dim = output_capsule_dim
        self.output_capsule_num = output_capsule_num

    def forward(self, poses):
        input_capsule_num = poses.shape[-2]
        u_hat_vecs = F.conv1d(poses.permute(0, 2, 1), self.kernel).view(
            (-1, input_capsule_num, self.output_capsule_num, self.output_capsule_dim))

        SHAPE = u_hat_vecs.permute((0, 2, 1, 3)).shape
        u_hat_vecs = u_hat_vecs.reshape((SHAPE[0], SHAPE[1], SHAPE[3], SHAPE[2]))
        u_hat_vecs = u_hat_vecs.permute((0, 1, 3, 2))
        return u_hat_vecs

class CapsTreeLayer(Module):
    def __init__(self, capsule_num, capsule_dim, num_node_type=100, routing_iteration=3,
                 nonlinear=squash, dropout=0.):
        super(CapsTreeLayer, self).__init__()
        self.pose_shape = capsule_dim
        self.labeldict = dict()

        self.fc:ModuleList = ModuleList()
        for _ in range(num_node_type):
            # output_capsule_num, poses_shape, iterations = 3, nonlinear = squash, dropout = 0.
            self.fc.append(capsule_fc_layer(
                output_capsule_num=capsule_num, iterations=routing_iteration,
                poses_shape=self.pose_shape, nonlinear=nonlinear, dropout=dropout
            ))

    def isLeaf(self, node):
        if isinstance(node, str): return True
        else:                     return False

    def label2idx(self, label):
        if label not in self.labeldict:
            self.labeldict[label] = len(self.labeldict)
        return self.labeldict[label]
        # return 0

    def fusion(self, inputs:list) -> torch.Tensor or None:
        if len(inputs) == 0:
            return None
        sums = torch.sum(torch.stack(inputs), dim=0)
        avg = torch.div(sums, len(inputs))
        return avg

    def dfs(self, node, pose, tree, mapdict=None):
        # print(node.label())
        cache = []
        for child in node:
            if self.isLeaf(child):
                idx = tree.leaves().index(child)
                idx = mapdict[idx] if mapdict else idx
                if idx >= pose.shape[1]:        # 由于设置maxlength，而造成了树的裁剪
                    continue
                cache.append( pose[:, idx] )
                # 如果要 “词的init_caps" 就留着下面这一行，否则注释掉
                # self.node_features.append(pose[:, idx])
            else:
                idx = self.label2idx(child.label())
                out = self.dfs(child, pose, tree, mapdict)
                if out is None:   continue
                ans = self.fc[idx](out)
                cache.append(ans)
                self.node_features.append(ans)
                # self.node_features.append(ans / (node.height() - 1))
        return self.fusion(cache)

    def forward(self, net, tree, mapdict=None):
        # print(self.fc[0].vec_transformation.kernel[0, 1, 1])
        pose = net
        pose = pose.squeeze(2)
        self.node_features = []
        output = self.dfs(tree, pose, tree, mapdict)
        # return output
        self.node_features.append(output)
        self.node_features = [i for i in self.node_features if i is not None]
        return torch.stack(self.node_features, dim=1) if len(self.node_features) != 0 else None
