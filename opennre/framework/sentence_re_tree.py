import torch
from torch import nn, optim
from nltk.tree import Tree
import json
from .data_loader import SentenceRELoader
from .utils import AverageMeter
from tqdm import tqdm
import os
from .scorer import score

class SentenceREwithTree(nn.Module):

    def __init__(self, 
                 model,
                 train_path, 
                 val_path, 
                 test_path,
                 ckpt, 
                 batch_size=32, 
                 max_epoch=100, 
                 lr=0.1, 
                 weight_decay=1e-5, 
                 opt='sgd'):

        super().__init__()
        self.max_epoch = max_epoch
        # Load data
        if train_path != None:
            self.train_loader = SentenceRELoader(
                train_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                shuffle = True,
                USETREE=True,
                USEPOS_NER='tacred' in ckpt and 'bert' not in ckpt,
            )

        if val_path != None:
            self.val_loader = SentenceRELoader(
                val_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                shuffle = False,
                USETREE=True,
                USEPOS_NER='tacred' in ckpt and 'bert' not in ckpt,
            )

        if test_path != None:
            self.test_loader = SentenceRELoader(
                test_path,
                model.rel2id,
                model.sentence_encoder.tokenize,
                batch_size,
                shuffle = False,
                USETREE = True,
                USEPOS_NER='tacred' in ckpt and 'bert' not in ckpt,
            )
        # Model
        self.model = model
        # self.parallel_model = nn.DataParallel(self.model)
        self.parallel_model = self.model
        # Criterion
        self.criterion = nn.CrossEntropyLoss()
        # Params and optimizer
        # params = self.parameters()
        params = filter(lambda p: p.requires_grad, self.parameters())
        self.lr = lr
        self.weight_decay = weight_decay
        self.opt = opt
        self.group_train = False
        if opt == 'sgd':
            self.optimizer = optim.SGD(params, lr, weight_decay=weight_decay)
        elif opt == 'adam':
            self.optimizer = optim.Adam(params, lr, weight_decay=weight_decay)
        elif opt == 'adadelta':
            self.optimizer = optim.Adadelta(params, lr, weight_decay=weight_decay)
        elif opt == 'adamw': # Optimizer for BERT
            from transformers import AdamW
            params = list(self.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            grouped_params = [
                {
                    'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 
                    'weight_decay': 0.01,
                    'lr': lr,
                    'ori_lr': lr
                },
                {
                    'params': [p for n, p in params if any(nd in n for nd in no_decay)], 
                    'weight_decay': 0.0,
                    'lr': lr,
                    'ori_lr': lr
                }
            ]
            self.optimizer = AdamW(grouped_params, correct_bias=False)
        else:
            raise Exception("Invalid optimizer. Must be 'sgd' or 'adam' or 'adamw'.")
        # Cuda
        if torch.cuda.is_available():
            self.cuda()
        # Ckpt
        self.ckpt = ckpt

    def train_model(self, warmup=True, metric='acc'):
        best_metric = 0
        global_step = 0

        for epoch in range(self.max_epoch):
            self.train()
            print("=== Epoch %d train ===" % epoch)
            avg_loss = AverageMeter()
            avg_acc = AverageMeter()

            t = tqdm(self.train_loader)
            for iter, data in enumerate(t):
                for i in range(len(data)):
                    try:
                        data[i] = data[i].cuda()
                    except:
                        pass
                label = data[0]
                args = data[1:]
                try:
                    logits = self.parallel_model(*args)
                except Exception as e:
                    print("Error: ", e)
                    logits = self.parallel_model(*args)
                    continue
                loss = self.criterion(logits, label)
                score, pred = logits.max(-1) # (B)
                acc = float((pred == label).long().sum()) / label.size(0)
                avg_loss.update(loss.item(), 1)
                avg_acc.update(acc, 1)
                t.set_postfix(loss=avg_loss.avg, acc=avg_acc.avg)
                # Optimize
                if warmup == True:
                    warmup_step = 300
                    if global_step < warmup_step:
                        warmup_rate = float(global_step) / warmup_step
                    else:
                        warmup_rate = 1.0
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.lr * warmup_rate

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                global_step += 1

                if iter == len(self.train_loader) // 2 and 'tacred' in self.ckpt.lower():
                    print("=== Epoch %d val ===" % epoch)
                    if 'semeval' in self.ckpt:
                        result = self.eval_model(self.val_loader, metric='semeval')
                    elif 'tacred' in self.ckpt.lower():
                        result = self.eval_model(self.val_loader, metric='tacred')
                    else:
                        result = self.eval_model(self.val_loader)
                    print(metric, ':', result[metric])
                    if result[metric] > best_metric:
                        print("Best ckpt and saved.")
                        folder_path = '/'.join(self.ckpt.split('/')[:-1])
                        if not os.path.exists(folder_path):
                            os.mkdir(folder_path)
                        torch.save({'state_dict': self.model.state_dict()}, self.ckpt)
                        best_metric = result[metric]
            # Val 
            print("=== Epoch %d val ===" % epoch)
            if 'semeval' in self.ckpt:
                result = self.eval_model(self.val_loader, metric='semeval')
            elif 'tacred' in self.ckpt.lower():
                result = self.eval_model(self.val_loader, metric='tacred')
            else:
                result = self.eval_model(self.val_loader)
            print(metric, ':', result[metric])
            if result[metric] > best_metric:
                print("Best ckpt and saved.")
                folder_path = '/'.join(self.ckpt.split('/')[:-1])
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)
                torch.save({'state_dict': self.model.state_dict()}, self.ckpt)
                best_metric = result[metric]
        print("Best %s on val set: %f" % (metric, best_metric))

    def eval_model(self, eval_loader, metric='default'):
        self.eval()
        if metric == 'default':
            avg_acc = AverageMeter()
            pred_result = []
            with torch.no_grad():
                t = tqdm(eval_loader)
                for iter, data in enumerate(t):
                    if torch.cuda.is_available():
                        for i in range(len(data)):
                            try:
                                data[i] = data[i].cuda()
                            except:
                                pass
                    label = data[0]
                    args = data[1:]
                    logits = self.parallel_model(*args)
                    score, pred = logits.max(-1) # (B)
                    # Save result
                    for i in range(pred.size(0)):
                        pred_result.append(pred[i].item())
                    # Log
                    acc = float((pred == label).long().sum()) / label.size(0)
                    avg_acc.update(acc, pred.size(0))
                    t.set_postfix(acc=avg_acc.avg)
            result = eval_loader.dataset.eval(pred_result)
            return result
        elif metric == 'semeval':
            # raise NotImplementedError
            avg_acc = AverageMeter()
            pred_result = []
            preds, labels = [], []
            with torch.no_grad():
                t = tqdm(eval_loader)
                for iter, data in enumerate(t):
                    if torch.cuda.is_available():
                        for i in range(len(data)):
                            try:
                                data[i] = data[i].cuda()
                            except:
                                pass
                    label, args = data[0], data[1:]
                    logits = self.parallel_model(*args)
                    score, pred = logits.max(-1) # (B)
                    preds.append(pred)
                    labels.append(label)
                    # Save result
                    for i in range(pred.size(0)):
                        pred_result.append(pred[i].item())
                    # Log
                    acc = float((pred == label).long().sum()) / label.size(0)
                    avg_acc.update(acc, pred.size(0))
                    t.set_postfix(acc=avg_acc.avg)
            result = eval_loader.dataset.eval(pred_result)
            result.update(self.F1_score_macro(
                torch.cat(preds, dim=0), torch.cat(labels, dim=0), logits.shape[1]))
            return result
        elif metric == 'tacred':
            # raise NotImplementedError
            avg_acc = AverageMeter()
            pred_result = []
            preds, labels = [], []
            with torch.no_grad():
                t = tqdm(eval_loader)
                for iter, data in enumerate(t):
                    if torch.cuda.is_available():
                        for i in range(len(data)):
                            try:
                                data[i] = data[i].cuda()
                            except:
                                pass
                    label, args = data[0], data[1:]
                    logits = self.parallel_model(*args)
                    score, pred = logits.max(-1) # (B)
                    preds.append(pred)
                    labels.append(label)
                    # Save result
                    for i in range(pred.size(0)):
                        pred_result.append(pred[i].item())
                    # Log
                    acc = float((pred == label).long().sum()) / label.size(0)
                    avg_acc.update(acc, pred.size(0))
                    t.set_postfix(acc=avg_acc.avg)
            result = eval_loader.dataset.eval(pred_result)
            # pred, label, class_num = torch.cat(preds, dim=0), torch.cat(labels, dim=0), logits.shape[1]
            result.update(self.F1_score_micro_tacre(
                torch.cat(preds, dim=0), torch.cat(labels, dim=0), logits.shape[1]))
            return result
        else:
            raise Exception("Unknown metric. ")

    def F1_score_micro_tacre(self, pred, label, class_num):
        label, pred = label.cpu().numpy().tolist(), pred.cpu().numpy().tolist()
        prec_micro, recall_micro, f1_micro = score(label, pred, False)
        ans = {
            'micro_f1_tacred': f1_micro,
            'micro_p_tacred': prec_micro,
            'micro_r_tacred': recall_micro,
        }
        return ans

    def F1_score_macro(self, pred, label, class_num, threshold=0.5, beta=1):
        E = torch.eye(class_num, device=pred.device) > 0
        prob, label = E[pred], E[label]

        # 合并为九类
        start = 1
        TP = (prob & label).sum(dim=0).float()[start: start + 18].view(-1, 2).sum(dim=1)
        TN = ((~prob) & (~label)).sum(dim=0).float()[start: start + 18].view(-1, 2).sum(dim=1)
        FP = (prob & (~label)).sum(dim=0).float()[start: start + 18].view(-1, 2).sum(dim=1)
        FN = ((~prob) & label).sum(dim=0).float()[start: start + 18].view(-1, 2).sum(dim=1)

        precision = TP / (TP + FP + 1e-19)
        recall = TP / (TP + FN + 1e-19)
        F2 = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall + 1e-19)
        f1 = F2.mean(0)
        ans = {
            'macro_f1': f1.item(),
        }
        return ans

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

