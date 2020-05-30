#!/user/bin/env python
# -*- coding: utf-8 -*-

from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import json
from tqdm import tqdm
from collections import defaultdict
from nltk.tree import Tree

def get_length(d):
    return len(d['token'])

def get_label(d):
    return d['relation']

def get_depth(d):
    return Tree.fromstring(d['tree']).height()

def report(over=None):
    no_relation = save_labels.get('Other', 0) + save_labels.get('no_relation', 0)
    no_relation_rate = no_relation / sum(save_labels.values())
    print(" Relations:  {} (no_rel: {}%)".format(len(save_labels), round(no_relation_rate*100, 1)))
    # print(list(save_labels.keys()))
    print(" Length:     Max: {}  Avg: {}  Min: {}".format(
        max(lengths), round(sum(lengths) / len(lengths), 1), min(lengths)))
    print(" Depth:      Max: {}  Avg: {}  Min: {}".format(
        max(depths), round(sum(depths) / len(depths), 1), min(depths)))
    if over:
        print("{} / {} 's Length is more than {}.".format(len([l for l in lengths if l >= over]),
                                                          len(lengths),
                                                          over))




detail = False

# dataset = 'semeval'
dataset = 'tacred'




lengths, save_labels, depths = [], defaultdict(int), []
for split in ['train', 'val', 'test']:
    file = "./{0}/{0}_{1}.txt".format(dataset, split)
    # file = "./{0}/raw/{0}_{1}.txt".format(dataset, split)
    if detail:
        print("\n{}:".format(split))
        lengths, save_labels, depths = [], defaultdict(int), []
    with open(file, 'r', encoding='utf8') as f:
        for line in tqdm(f):
            d = json.loads(line.strip())
            length = get_length(d)
            label = get_label(d)
            depth = get_depth(d)

            save_labels[label] += 1
            lengths.append(length)
            depths.append(depth)
    if detail:
        report()

if not detail:
    print("\nAll:")
    report(over=74)
