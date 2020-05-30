#!/user/bin/env python
# -*- coding: utf-8 -*-
import os, torch, json
import numpy as np
from tqdm import tqdm
from nltk.tree import Tree
from torch import nn
import opennre
from opennre import encoder, model, framework
from opennre.framework.data_loader import SentenceRELoader
import subprocess

CASE_STUDY = False

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

MODEL_NAME = 'semeval_ablstm_adadelta_softmax'
MODEL_NAME = 'semeval_ablstm_adadelta_nopad_softmax'
MODEL_NAME = 'semeval_tree_softmax.pth.tar'
MODEL_NAME = 'semeval_caps+lstm_softmax.pth.tar' # P =  78.47%	R =  81.66%	F1 =  79.92%
# MODEL_NAME = 'semeval_capsNoPre8005_softmax.pth'   # P =  78.06%	R =  82.38%	F1 =  80.05%
# MODEL_NAME = 'semeval_capsNoPre8048_softmax.pth'   # P =  78.26%	R =  83.06%	F1 =  80.48%
MODEL_NAME = 'semeval_capsNoPre_softmax.pth'   # P =  78.50%	R =  82.58%	F1 =  80.39%
# # MODEL_NAME = 'semeval_tree_softmax.pth.tar'   #
# MODEL_NAME = 'semeval_tree_fc_softmax.pth.tar'   # P =  76.51%	R =  82.74%	F1 =  79.36%
MODEL_NAME = 'semeval_tree_fcNoPre_softmax.pth.tar'   #pycharm P =  77.06%	R =  83.57%	F1 =  80.05%
MODEL_NAME = 'semeval_tree_PrecapAda01.pth.tar'   #5 7931 P =  77.77%	R =  82.89%	F1 =  80.09%  # P =  78.24%	R =  82.90%	F1 =  80.37%
# MODEL_NAME = 'semeval_tree_PrecapAda01Treedecay_softmax.pth.tar'   #1 7957 P =  77.56%	R =  82.82%	F1 =  79.95% # P =  78.05%	R =  82.83%	F1 =  80.23%
# MODEL_NAME = 'semeval_tree_PrecapAda01FT_softmax.pth.tar'   #2 7966 P =  76.75%	R =  83.35%	F1 =  79.81%

MODEL_NAME = 'semeval_cnntest_softmax.pth.tar'   # P =  80.43%	R =  76.92%	F1 =  78.54%

MODEL_NAME = 'semeval_lstmCat_91.pth.tar'   # P =  78.57%	R =  81.67%	F1 =  79.99%


#                                        # backBone: 80.74%
# MODEL_NAME = 'semeval_tree_cat'        # P =  79.71%	R =  81.78%	F1 =  80.60%
# MODEL_NAME = 'semeval_tree_cat2'       # P =  79.62%	R =  81.91%	F1 =  80.62%
# MODEL_NAME = 'semeval_tree_cat3'       # P =  79.81%	R =  81.32%	F1 =  80.42%
#
MODEL_NAME = 'semeval_tree_cat5'       #

MODEL_NAME = 'tacred_cnn1'       #
# MODEL_NAME = 'tacred_lstm03'       #
#
MODEL_NAME = 'semeval_lstmpa'       #
MODEL_NAME = 'tacred_lstmpa'       #

MODEL_NAME = 'semeval_tree_cat_withoutLSTM1'
# MODEL_NAME = 'semeval_tlpre'       #
# MODEL_NAME = 'semeval_tl'       #


dataset = MODEL_NAME.split('_')[0]

def loadModel(ckpt):
    ckpt = ckpt.replace(".pth", "")
    ckpt = ckpt.replace(".tar", "")
    root_path = os.path.curdir
    dataset = ckpt.split('_')[0]

    if dataset == 'semeval':
        wordEmbedding = 'glove/glove.6B.50d'
    elif dataset == 'tacred':
        wordEmbedding = 'glove/glove.840B.300d'
    # wordEmbedding = 'turian/turian.50d'
    word2id = json.load(open('pretrain/{}_word2id.json'.format(wordEmbedding)))
    word2vec = np.load('pretrain/{}_mat.npy'.format(wordEmbedding))
    rel2id = json.load(open(os.path.join(root_path, 'benchmark/{0}/{0}_rel2id.json'.format(dataset))))
    if '_cnn' in ckpt:
        sentence_encoder = opennre.encoder.CNNEncoder(token2id=word2id,
                                                      max_length=100,
                                                      word_size={"semeval": 50, "tacred": 300}[dataset],
                                                      position_size={"semeval": 5, "tacred": 30}[dataset],
                                                      pos_ner=(dataset == 'tacred'),
                                                      hidden_size={"semeval": 100, "tacred": 500}[dataset],
                                                      blank_padding=True,
                                                      kernel_size={"semeval": [3, 4, 5], "tacred": [2, 3, 4, 5]}[
                                                          dataset],
                                                      padding_size={"semeval": [1, 2, 3], "tacred": [1, 2, 3, 4]}[
                                                          dataset],
                                                      word2vec=word2vec,
                                                      dropout={"semeval": 0.5, "tacred": 0.3}[dataset], )
        m = model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
    elif '_tl' in ckpt:
        sentence_encoder = opennre.encoder.TreeLSTMEncoder(token2id=word2id,
                                                           num_class=len(rel2id),
                                                           max_length=100,
                                                           word_size={"semeval": 50, "tacred": 300}[dataset],
                                                           position_size={"semeval": 5, "tacred": 30}[dataset],
                                                           pos_ner=(dataset == 'tacred'),
                                                           blank_padding=True,
                                                           word2vec=word2vec,
                                                           emb_dropout={"semeval": 0.3, "tacred": 0.5}[dataset],
                                                           rnn_dropout={"semeval": 0.5, "tacred": 0.5}[dataset],
                                                           hidden_size=256,
                                                           # pretrainLSTM=pretrainLSTM,
                                                           pretrainLSTM=None,
                                                           # fineTune = True,
                                                           fineTune=False,
                                                           )
        m = opennre.model.SoftmaxNNwofc(sentence_encoder, rel2id)

    elif '_tree' in ckpt:
        # sentence_encoder = opennre.encoder.CapTreeEncoder(token2id=wordi2d,
        sentence_encoder = opennre.encoder.CapTreeCatEncoder(token2id=word2id,
                                                          num_class=len(rel2id),
                                                          max_length=100,
                                                          word_size={"semeval": 50, "tacred": 300}[dataset],
                                                          position_size={"semeval": 5, "tacred": 30}[dataset],
                                                          pos_ner=(dataset == 'tacred'),
                                                          blank_padding=True,
                                                          word2vec=word2vec,
                                                          emb_dropout=0.5,
                                                          rnn_dropout=0.5,
                                                          cap_dropout=0.5,
                                                          hidden_capsule_num=16,
                                                          capsule_size=16,
                                                          )
        m = opennre.model.SoftmaxNNwofc(sentence_encoder, rel2id)
    elif '_cap' in ckpt:
        sentence_encoder = opennre.encoder.CapsuleEncoder(token2id=word2id,
                                                          num_class=len(rel2id),
                                                          max_length=100,
                                                          word_size={"semeval": 50, "tacred": 300}[dataset],
                                                          position_size={"semeval": 5, "tacred": 30}[dataset],
                                                          pos_ner=(dataset == 'tacred'),
                                                          blank_padding=True,
                                                          word2vec=word2vec,
                                                          emb_dropout=0.3,
                                                          rnn_dropout=0.3,
                                                          hidden_capsule_num=16,
                                                          capsule_size=16,
                                                          pretrainLSTM=None,
                                                          )
        m = model.SoftmaxNNwofc(sentence_encoder, rel2id)
    elif '_lstmpa' in ckpt and 'cap' not in ckpt and 'tree' not in ckpt:
        sentence_encoder = opennre.encoder.PALSTMEncoder(token2id=word2id,
        # sentence_encoder = opennre.encoder.LSTMEncoder(token2id=wordi2d,
                                                            max_length=100,
                                                            word_size={"semeval": 50, "tacred": 300}[dataset],
                                                            position_size={"semeval": 5, "tacred": 30}[dataset],
                                                            pos_ner=(dataset == 'tacred'),
                                                            hidden_size=256,
                                                            blank_padding=True,
                                                            word2vec=word2vec,
                                                            emb_dropout=0.5,
                                                            rnn_dropout=0.5)
        m = model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
    elif ('_lstm' in ckpt or '_ablstm' in ckpt) and 'cap' not in ckpt and 'tree' not in ckpt:
        sentence_encoder = opennre.encoder.LSTMEntityEncoder(token2id=word2id,
        # sentence_encoder = opennre.encoder.LSTMEncoder(token2id=wordi2d,
                                                            max_length=100,
                                                            word_size={"semeval": 50, "tacred": 300}[dataset],
                                                            position_size={"semeval": 5, "tacred": 30}[dataset],
                                                            pos_ner=(dataset == 'tacred'),
                                                            hidden_size=256,
                                                            blank_padding=True,
                                                            word2vec=word2vec,
                                                            emb_dropout=0.5,
                                                            rnn_dropout=0.5)
        m = model.SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
    else:
        raise Exception("Unknown model. ")

    filePath = os.path.join(root_path, 'ckpt', ckpt + '.pth.tar')
    m.load_state_dict(torch.load(filePath)['state_dict'])
    return m

def case_study(logits, lstm_logits, label):
#    self.parallel_model(*args)   self.parallel_model.logit_lstm  ,  self.parallel_model.logit_capsule
    score1, pred1 = logits.max(-1)  # (B)
    # acc1 = float((pred1 == label).long().sum()) / label.size(0)

    score2, pred2 = lstm_logits.max(-1)  # (B)
    # acc2 = float((pred2 == label).long().sum()) / label.size(0)

    select_index = ((pred1 == label) & (pred2 != label))
    select_index = select_index.to_sparse()._indices()[0]
    return select_index, pred1, score1, pred2, score2

def getPredict(model, test_path, useTree=False, usePOS_NER=False):
    test_loader = SentenceRELoader(
        test_path,
        model.rel2id,
        model.sentence_encoder.tokenize,
        25,
        False,
        USETREE=useTree,
        USEPOS_NER=usePOS_NER,
    )
    model.eval()
    pred_result, Y = [], []
    with torch.no_grad():
        t = tqdm(test_loader)
        for iter, data in enumerate(t):
            # CASE_STUDY
            if CASE_STUDY:
                if iter < 30:
                    continue


            if torch.cuda.is_available():
                for i in range(len(data)):
                    try:
                        data[i] = data[i].cuda()
                    except:
                        pass
            label = data[0]
            args = data[1:]
            # logits = nn.DataParallel(model)(*args)
            logits = model(*args)
            score, pred = logits.max(-1)  # (B)

            # CASE_STUDY
            if CASE_STUDY:
                logit1, logit2, logit3 = logits, model.sentence_encoder.logit_lstm, model.sentence_encoder.logit_capsule
                select_index, p1, s1, p2, s2 = case_study(logit1, logit2, label)
                if select_index.shape[0] > 0:
                    inputs_data = [i[select_index] for i in data[:-1]]
                    inputs_data.append([data[-1][t] for t in select_index])
                    t = Tree.fromstring(data[4][select_index])
                    l = label[select_index]
                    p1 = p1[select_index]
                    p2 = p2[select_index]
                    modelpred = torch.softmax(logit1[select_index], dim=1).tolist()
                    lstmpred = torch.softmax(logit2[select_index], dim=1).tolist()
                    capspred = torch.softmax(logit3[select_index], dim=1).tolist()
                    capspred = logit3[select_index].tolist()
                    _ = model(*inputs_data[1:])

            # Save result
            for i in range(pred.size(0)):
                pred_result.append(pred[i].item())
                Y.append(label[i].item())
    return pred_result, Y

def semeval(MODEL_NAME):
    label2class = {0: 'Other',
                   1: 'Message-Topic(e1,e2)', 2: 'Message-Topic(e2,e1)',
                   3: 'Product-Producer(e1,e2)', 4: 'Product-Producer(e2,e1)',
                   5: 'Instrument-Agency(e1,e2)', 6: 'Instrument-Agency(e2,e1)',
                   7: 'Entity-Destination(e1,e2)', 8: 'Entity-Destination(e2,e1)',
                   9: 'Cause-Effect(e1,e2)', 10: 'Cause-Effect(e2,e1)',
                   11: 'Component-Whole(e1,e2)', 12: 'Component-Whole(e2,e1)',
                   13: 'Entity-Origin(e1,e2)', 14: 'Entity-Origin(e2,e1)',
                   15: 'Member-Collection(e1,e2)', 16: 'Member-Collection(e2,e1)',
                   17: 'Content-Container(e1,e2)', 18: 'Content-Container(e2,e1)'}

    model = loadModel(MODEL_NAME)
    try:
        model.cuda()
    except:
        pass
    pred, Y = getPredict(model, './benchmark/semeval/semeval_test.txt', 'tree' in MODEL_NAME or 'tl' in MODEL_NAME)

    preds, truths = np.array(pred), np.array(Y)
    prediction_path = "./ckpt/predictions.txt"
    truth_path ="./ckpt/ground_truths.txt"
    prediction_file = open(prediction_path, 'w')
    truth_file = open(truth_path, 'w')
    for i in range(len(preds)):
        # if preds[i] not in label2class:    p = 'Other'
        # else:                              p = label2class[preds[i]]
        prediction_file.write("{}\t{}\n".format(i, label2class[preds[i]]))
        truth_file.write("{}\t{}\n".format(i, label2class[truths[i]]))
    prediction_file.close()
    truth_file.close()

    perl_path = os.path.join("/home/ytc/CapsTree/attBiLSTM/",
                             "SemEval2010_task8_all_data",
                             "SemEval2010_task8_scorer-v1.2",
                             "semeval2010_task8_scorer-v1.2.pl")
    process = subprocess.Popen(["perl", perl_path, prediction_path, truth_path], stdout=subprocess.PIPE)
    for line in str(process.communicate()[0].decode("utf-8")).split("\\n"):
        print(line)

def tacred(MODEL_NAME):
    test_path = './benchmark/tacred/tacred_test.txt'
    model = loadModel(MODEL_NAME)
    try:
        model.cuda()
    except:
        pass

    dataset = 'tacred'
    framework = opennre.framework.SentenceRE(
        train_path='benchmark/{0}/{0}_train.txt'.format(dataset),
        val_path='benchmark/{0}/{0}_val.txt'.format(dataset),
        test_path='benchmark/{0}/{0}_test.txt'.format(dataset),
        model=model,
        ckpt='tacred',
        batch_size={"semeval": 32, "tacred": 128}[dataset],
        max_epoch=100,
        lr=0.1,
        weight_decay=1e-5,
        opt={"semeval": 'sgd', "tacred": 'adagrad'}[dataset])

    result = framework.eval_model(framework.test_loader, metric='tacred')
    print('Accuracy on test set: {}'.format(result['acc']))
    print('Micro Precision: {}'.format(result['micro_p']))
    print('Micro Recall: {}'.format(result['micro_r']))
    print('Micro F1: {}'.format(result['micro_f1']))
    # print('Macro F1: {}'.format(result['macro_f1']))

    print('micro_f1_tacred: {}'.format(result['micro_f1_tacred']))
    print('micro_p_tacred: {}'.format(result['micro_p_tacred']))
    print('micro_r_tacred: {}'.format(result['micro_r_tacred']))


if __name__ == '__main__':
    if 'semeval' in MODEL_NAME:
        semeval(MODEL_NAME)
    elif 'tacred' in MODEL_NAME:
        tacred(MODEL_NAME)
    else:
        raise Exception("Unknown dataset. ")
