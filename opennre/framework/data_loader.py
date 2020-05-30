import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json
import sklearn.metrics

# USETREE = False

class SentenceREDataset(data.Dataset):
    """
    Sentence-level relation extraction dataset
    """
    def __init__(self, path, rel2id, tokenizer, kwargs):
        """
        Args:
            path: path of the input file
            rel2id: dictionary of relation->id mapping
            tokenizer: function of tokenizing
        """
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.kwargs = kwargs

        # Load the file
        f = open(path)
        self.data = []
        for line in f.readlines():
            line = line.rstrip()
            if len(line) > 0 and len(json.loads(line)['token']) < 74:
                # drop samples whose tokens are more than 74.
                self.data.append(eval(line))
        f.close()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        seq = list(self.tokenizer(item, **self.kwargs))
        res = [self.rel2id[item['relation']]] + seq
        return [self.rel2id[item['relation']]] + seq # label, seq1, seq2, ...
    
    def collate_fn(data):
        data = list(zip(*data))
        labels = data[0]
        seqs = data[1:]
        # seqs = data[1:4]
        batch_labels = torch.tensor(labels).long() # (B)
        batch_seqs = []
        for seq in seqs:
            try:
                batch_seqs.append(torch.cat(seq, 0)) # (B, L)
            except:
                batch_seqs.append(seq)

        return [batch_labels] + batch_seqs

    def collate_fn_tree(data):
        data = list(zip(*data))
        labels = data[0]
        # seqs = data[1:]
        seqs = data[1:-1]
        batch_labels = torch.tensor(labels).long() # (B)
        batch_seqs = []
        for seq in seqs:
            try:
                batch_seqs.append(torch.cat(seq, 0)) # (B, L)
            except:
                batch_seqs.append(seq)

        tree = data[-1]
        return [batch_labels] + batch_seqs + [tree]

    def collate_fn_tree_pos(data):
        data = list(zip(*data))
        labels = data[0]
        # seqs = data[1:]
        seqs = data[1:-3]
        batch_labels = torch.tensor(labels).long() # (B)
        batch_seqs = []
        for seq in seqs:
            try:
                batch_seqs.append(torch.cat(seq, 0)) # (B, L)
            except:
                batch_seqs.append(seq)
        tree = data[-3]
        PosNer = []
        for seq in data[-2:]:
            PosNer.append(torch.cat(seq, 0)) # (B, L)
        return [batch_labels] + batch_seqs + [tree] + PosNer
    
    def eval(self, pred_result, use_name=False):
        """
        Args:
            pred_result: a list of predicted label (id)
                Make sure that the `shuffle` param is set to `False` when getting the loader.
            use_name: if True, `pred_result` contains predicted relation names instead of ids
        Return:
            {'acc': xx}
        """
        correct = 0
        total = len(self.data)
        correct_positive = 0
        pred_positive = 0
        gold_positive = 0
        neg = -1
        for name in ['NA', 'na', 'no_relation', 'Other', 'Others']:
            if name in self.rel2id:
                if use_name:
                    neg = name
                else:
                    neg = self.rel2id[name]
                break
        for i in range(total):
            if use_name:
                golden = self.data[i]['relation']
            else:
                golden = self.rel2id[self.data[i]['relation']]
            if golden == pred_result[i]:
                correct += 1
                if golden != neg:
                    correct_positive += 1
            if golden != neg:
                gold_positive +=1
            if pred_result[i] != neg:
                pred_positive += 1
        acc = float(correct) / max(float(total), 1e-19)
        micro_p = float(correct_positive) / max(float(pred_positive), 1e-19)
        micro_r = float(correct_positive) / max(float(gold_positive), 1e-19)
        micro_f1 = 2 * micro_p * micro_r / max((micro_p + micro_r), 1e-19)
        return {'acc': acc, 'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1}
    
def SentenceRELoader(path, rel2id, tokenizer, batch_size, 
        shuffle, num_workers=8, collate_fn=SentenceREDataset.collate_fn,
        USETREE=False, USEPOS_NER=False, **kwargs):
    if USETREE and not USEPOS_NER: collate_fn=SentenceREDataset.collate_fn_tree
    if USEPOS_NER: collate_fn=SentenceREDataset.collate_fn_tree_pos

    dataset = SentenceREDataset(path = path, rel2id = rel2id, tokenizer = tokenizer, kwargs=kwargs)
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return data_loader

