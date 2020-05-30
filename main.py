#!/user/bin/env python
# -*- coding: utf-8 -*-

# coding:utf-8
import torch, os
import numpy as np
import json
import opennre
from opennre import encoder, model, framework
from print_state import PrintState
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


dataset = 'semeval'
# dataset = 'tacred'

if dataset == 'semeval':       # Represents the trained model of the first stage
    pretrainLSTM = './pretrain/semeval_lstmCat_8074.pth.tar'
else:
    pretrainLSTM = './pretrain/tacred_lstmpa.pth.tar'

ckpt = 'ckpt/{}.pth.tar'.format(dataset)
print(ckpt)

wordEmbedding = 'glove/glove.6B.50d' if dataset == 'semeval' else 'glove/glove.840B.300d'
word2id = json.load(open('pretrain/{}_word2id.json'.format(wordEmbedding)))
word2vec = np.load('pretrain/{}_mat.npy'.format(wordEmbedding))
rel2id = json.load(open('benchmark/{0}/{0}_rel2id.json'.format(dataset)))

sentence_encoder = opennre.encoder.CapTreeCatEncoder(token2id = word2id,
                    num_class = len(rel2id),
                    max_length = 100,
                    word_size={"semeval": 50, "tacred": 300}[dataset],
                    position_size={"semeval": 5, "tacred": 30}[dataset],
                    pos_ner= (dataset=='tacred'),
                    blank_padding = True,
                    word2vec = word2vec,
                    emb_dropout={"semeval": 0.5, "tacred": 0.5}[dataset],
                    rnn_dropout={"semeval": 0.5, "tacred": 0.5}[dataset],
                    cap_dropout = 0.5,
                    hidden_capsule_num=16,
                    capsule_size = 16,
                    pretrainLSTM = pretrainLSTM,
                    fineTune = True,
                    gamma = 0.4,
                    )
model = opennre.model.SoftmaxNNwofc(sentence_encoder, rel2id)

framework = opennre.framework.SentenceREwithTree(
    train_path='benchmark/{0}/{0}_train.txt'.format(dataset),
    val_path='benchmark/{0}/{0}_val.txt'.format(dataset),
    test_path='benchmark/{0}/{0}_test.txt'.format(dataset),
    model=model,
    ckpt=ckpt,
    batch_size=32,
    max_epoch=200,
    lr=0.1,
    weight_decay=1e-5,
    opt={"semeval": 'adadelta', "tacred": 'adadelta'}[dataset],
)

print(PrintState(sentence_encoder, framework))

# Train
framework.train_model(metric={"semeval": "macro_f1", "tacred": "micro_f1"}[dataset], warmup=True)
# Test
framework.load_state_dict(torch.load(ckpt)['state_dict'])
result = framework.eval_model(framework.test_loader, metric={"semeval": "semeval", "tacred": "default"}[dataset])
print('Accuracy on test set: {}'.format(result['acc']))
print('Micro Precision: {}'.format(result['micro_p']))
print('Micro Recall: {}'.format(result['micro_r']))
print('Micro F1: {}'.format(result['micro_f1']))
print('Macro F1: {}'.format(result['macro_f1']))
