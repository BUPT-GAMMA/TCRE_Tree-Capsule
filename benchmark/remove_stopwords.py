#!/user/bin/env python
# -*- coding: utf-8 -*-

import json
from tqdm import tqdm
from nltk.tree import Tree

import nltk


class Discriminator(object):
    def __init__(self, dataset):
        self.stopwordsSet = set(nltk.corpus.stopwords.words('english'))
        self.brackets = {'semeval': list("()"), 'tacred': ['-LRB-', '-RRB-']}[dataset]
        self.bracket_flag = False
        self.bracket_delay = 0

    def stopwords(self, idx, word, pos1, pos2):
        '''
        Judge whether should be kept or not.
        :param idx: int:  word index. It is used to judge whether it is an entity.
        :param word: str:   word. It is used to judge whether it is stop word.
        :param pos1: list:  It is used to judge whether the word is an entity.
        :param pos2: list:  It is used to judge whether the word is an entity.
        :return: bool: If should be kept, True. Else False.
        '''
        # 该词是实体
        if pos1[0] <= idx < pos1[1] or pos2[0] <= idx < pos2[1]:
            return True
        # 该词不是停用词
        if word not in self.stopwordsSet:
            return True
        return False

    def bracket(self, idx, word, pos1, pos2):
        if pos1[0] <= idx < pos1[1] or pos2[0] <= idx < pos2[1]:
            if word in self.brackets:
                raise RuntimeError("A bracket is in Entity Name. ")
            return True
        if word == self.brackets[0]:
            self.bracket_flag = True
        elif word == self.brackets[1]:
            self.bracket_flag = False
            self.bracket_delay = 1

        if self.bracket_flag:
            return False
        elif self.bracket_delay > 0:
            self.bracket_delay -= 1
            return False
        else:
            return True


def remove_words(tokens: list, pos1: list, pos2: list, discriminator) -> tuple:
    res_token, deleted_idx = [], []
    for idx, word in enumerate(tokens):
        if discriminator(idx, word, pos1, pos2):
            res_token.append(word)
        else:
            deleted_idx.append(idx)

    # 处理Pos
    bias1, bias2 = len([i for i in deleted_idx if i < pos1[0]]), len([i for i in deleted_idx if i < pos2[0]])
    res_pos1 = [pos1[0] - bias1, pos1[1] - bias1]
    res_pos2 = [pos2[0] - bias2, pos2[1] - bias2]

    # 处理tree
    # t = Tree.fromstring(d['tree'])
    # if t.height() <= 7 and len(d['token']) <= 10 and d['relation'] != 'Other':
    # # if 'VBP' not in list(zip(*t.pos()))[1] and 'VB' not in list(zip(*t.pos()))[1]:
    #     t.pretty_print()
    #     print("here")


    return res_token, res_pos1, res_pos2


dataset = 'semeval'
dataset = 'tacred'   # 这个数据集不删了
discriminator = Discriminator(dataset)

for split in ['train', 'val', 'test']:
    file = "./{0}/{0}_{1}.txt".format(dataset, split)
    ans = []
    cnt = 1
    with open(file, 'r', encoding='utf8') as f:
        for line in tqdm(f):
            d = json.loads(line.strip())
            length = len(d['token'])
            # d['token'], d['h']['pos'], d['t']['pos'] = remove_words(
            #     d['token'], d['h']['pos'], d['t']['pos'], discriminator.stopwords)

            d['token'], d['h']['pos'], d['t']['pos'] = remove_words(
                d['token'], d['h']['pos'], d['t']['pos'], discriminator.bracket)

            try:
                assert " ".join(d['token'][d['h']['pos'][0]: d['h']['pos'][1]]), d['h']['name']
                assert " ".join(d['token'][d['t']['pos'][0]: d['t']['pos'][1]]), d['t']['name']
            except:
                print(cnt)

            # d['token'], d['h']['pos'], d['t']['pos'] = remove_bracket(
            #     d['token'], d['h']['pos'], d['t']['pos'])

            ans.append(json.dumps(d))
            cnt += 1

    with open(file, 'w', encoding='utf8') as f:
        f.write("\n".join(ans))