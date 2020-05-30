#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time, json, os
from multiprocessing import Pool as ProcessPool
from tqdm import tqdm
from nltk.parse import stanford
from nltk import sent_tokenize
time0 = time.time()

full_path = u"/home/ytc/nltk_data/stanford-parser-full-2018-02-27"  # 那三个文件的系统全路径目录
parser = stanford.StanfordParser(  # 加载解析器，注意：一定要是全路径，从系统根路径开始，不然找不到
    path_to_jar=full_path + u"/stanford-parser.jar",
    path_to_models_jar=full_path + u"/stanford-parser-3.9.1-models.jar",
    model_path=full_path + u'/jar/englishPCFG.ser.gz')
parser.java_options = '-mx2000m'

def getParser(text):
    text = text.replace("(", "[").replace(")", "]")

    res = list(parser.parse(text.split()))[0]
    return ' '.join(str(res).replace('\n', '').split(' '))
    # Load method:   nltk.tree.Tree.fromstring( string )

def run(para, parse_sentences=False):
    sentence = " ".join(para['token'])
    l = []
    if parse_sentences:
        for sent in sent_tokenize(sentence):
            try:
                res = getParser(sent)
                for i in range(10): res = res.replace("  ", " ")
                l.append( res )
            except:
                print("ERROR")
                print(sent)
                return run(para)
        l = "\t".join(l)
    else:
        sent = sentence
        try:
            res = getParser(sent)
            for i in range(10): res = res.replace("  ", " ")
            l.append( res )
        except:
            print("ERROR")
            print(sent)
            return run(para)
        l = "\t".join(l)
    ans = para
    ans['tree'] = l
    return ans

def process_pool(data):
    p = ProcessPool(40)
    res = list(tqdm(p.imap(run, data), total=len(data)))
    p.close()
    p.join()
    return res


def parse_tree(DATASET, out_file):
    ind = 0
    # for d in ['train', 'val', 'test']:
    for d in ['val', 'test']:
    # for d in ['test']:
        data = []
        data_file = './{0}/{0}_{1}.txt'.format(DATASET, d)
        with open(data_file, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        print(d, 'readed. Parsing...')
        
        outdata = process_pool(data)
        with open(data_file, 'w') as f:
            f.write("\n".join([json.dumps(od) for od in outdata]))


if __name__ == '__main__':
    DATASET = 'semeval'
    DATASET = 'tacred'
    parse_tree(DATASET, out_file=None)