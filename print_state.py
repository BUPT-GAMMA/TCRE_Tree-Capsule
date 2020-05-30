#!/user/bin/env python
# -*- coding: utf-8 -*-
import json

def PrintState(model, framework):
    d = {
        "max_length": model.max_length,
        "word_size": model.word_size,
        "position_size": model.position_size,
        "blank_padding": model.blank_padding,
        "emb_dropout": model.emb_dropout.p,
        "rnn_dropout": model.rnn_dropout.p,
        "cap_dropout": model.capsule_fc.dropout.dropout.p,
        "hidden_capsule_num": 16,
        "capsule_size": model.capsule_size,
        "fineTune": model.fine_tune,

        "lr": framework.lr,
        "weight_decay": framework.weight_decay,
        "opt": framework.opt,
    }
    return d
