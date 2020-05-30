#!/user/bin/env python
# -*- coding: utf-8 -*-

from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import json
from tqdm import tqdm


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatize_sentence(tokens: list) -> list:
    res = []
    lemmatizer = WordNetLemmatizer()
    for word, pos in pos_tag(tokens):
        wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
        res.append(lemmatizer.lemmatize(word, pos=wordnet_pos))
    return res

# s = " ".join(eval('["U.S.", "Immigration", "and", "Customs", "Enforcement", "agents", "seized", "documents", "and", "other", "materials", "at", "the", "Koch", "Foods", "plant", "and", "at", "Koch", "Foods", "Inc.", "\'s", "Chicago", "area", "headquarters", ",", "said", "Brian", "Moskowitz", ",", "a", "special", "agent", "in", "charge", "of", "ICE", "enforcement", "for", "Ohio", "and", "Michigan", "."]'))
# s = s.lower()
# print(s)
#
# print(lemmatize_sentence(s))


if __name__ == '__main__':

    dataset = 'semeval'

    for split in ['train', 'val', 'test']:
        file = "./{0}/{0}_{1}.txt".format(dataset, split)
        ans = []
        with open(file, 'r', encoding='utf8') as f:
            for line in tqdm(f):
                d = json.loads(line.strip())
                length = len(d['token'])
                d['token'] = lemmatize_sentence(d['token'])
                if len(d['token']) != length:
                    raise Exception("Different length.")
                ans.append(json.dumps(d))

        with open(file, 'w', encoding='utf8') as f:
            f.write("\n".join(ans))



