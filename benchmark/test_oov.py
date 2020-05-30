#!/user/bin/env python
# -*- coding: utf-8 -*-
import json
from tqdm import tqdm

word_embedding = 'glove'    # glove, turian, bert

dataset = 'semeval'

datadir = './{0}/无处理/{0}'.format(dataset)
# datadir = './{0}/{0}'.format(dataset)

wordi2d = json.load(open('../pretrain/glove/glove.6B.50d_word2id.json'))
wordi2d = json.load(open('../pretrain/turian/turian.50d_word2id.json'))


wordi2d = set(wordi2d.keys())

ans = {'all': [0, 0]}
voc = {'all': set()}
for split in ['train', 'val', 'test']:
    ans[split] = [0, 0]  # 有word2vec , OOV
    voc[split] = set()
    with open(datadir+"_{}.txt".format(split), 'r', encoding='utf8') as f:
        for line in tqdm(f):
            tokens = [t.lower() for t in json.loads(line.strip())['token']]
            voc[split].update(set(tokens))
            voc['all'].update(set(tokens))
            for t in tokens:
                if t in wordi2d:
                    ans[split][0] += 1
                    ans['all'][0] += 1
                else:
                    ans[split][1] += 1
                    ans['all'][1] += 1



for s in ans:
    cache = ans[s][0], sum(ans[s]), ans[s][1]
    print(s, "Token: {} / {} are in vocabulary, i.e., {} are OOV.".format(*cache))
    cache = len(voc[s] & wordi2d), len(voc[s])
    print(s, "Vocab: {} / {} are in vocabulary, i.e., {} are OOV.".format(cache[0], cache[1], cache[1]-cache[0]))
    print("")


