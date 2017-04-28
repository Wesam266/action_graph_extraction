# -*- coding: utf-8 -*-
'''
Code adapted from

'''

import numpy as np
import nltk
import pickle
from collections import Counter
from sklearn.naive_bayes import MultinomialNB




class OpSigModel:
    def __init__(self):
        self.reset()
        pass

    def save(self, fname):
        with open(fname, 'w') as f:
            pickle.dump((self.op_sig_cnts, self.op_sig_model_dict), f, pickle.HIGHEST_PROTOCOL)

    def load(self, fname):
        with open(fname, 'r') as f:
            self.op_sig_cnts, self.op_sig_model_dict = pickle.load(f)

    def reset(self):
        self.op_sig_cnts = np.array([0] * 8).astype(float)
        self.op_sig_model_dict = dict()

    def get_idx_from_opSig(self, sig_set, is_leaf):
        s = ''
        if 'DOBJ' in sig_set:
            s = s + '1'
        else:
            s = s + '0'

        if 'PP' in sig_set:
            s = s + '1'
        else:
            s = s + '0'

        if is_leaf:
            s = s + '1'
        else:
            s = s + '0'
        return int(s, 2)

    def get_opSig_idx(self, action):
        # is_leaf = True # set outside
        is_leaf = action.is_leaf

        sig_set = set()
        # get op sig
        for arg in action.ARGs:
            if arg.sem_type == 'material':
                sig_set = sig_set.union([arg.syn_type])

        return self.get_idx_from_opSig(sig_set, is_leaf)

    def get_opSig_from_idx(self, idx):
        bit_str = format(idx, '03b')
        assert len(bit_str) is 3

        sig_set = set()
        is_leaf = False
        if bit_str[0] is '1':
            sig_set = sig_set.union(['DOBJ'])

        if bit_str[1] is '1':
            sig_set = sig_set.union(['PP'])

        if bit_str[2] is '1':
            is_leaf = True

        return sig_set, is_leaf



    def norm_cnt(self, cnt):
        # make sure it's cnt not prob

        res = float(1) / len(cnt)  #
        cnt[cnt == 0] = res

        s = sum(cnt)
        # assert s != 0
        if s > 0:
            prob = cnt / s
            return prob

    def M_step(self, actionGraphs):
        self.reset()

        for AG in actionGraphs:
            for action in AG.actions:
                op = action.op
                idx = self.get_opSig_idx(action)
                self.op_sig_cnts[idx] = self.op_sig_cnts[idx] + 1

                if op not in self.op_sig_model_dict.keys():
                    self.op_sig_model_dict[op] =  np.array([0] * 8).astype(float)
                self.op_sig_model_dict[op][idx] = self.op_sig_model_dict[op][idx] + 1

        # normalize probs
        self.op_sig_cnts = self.norm_cnt(self.op_sig_cnts)
        for k in self.op_sig_model_dict.keys():
            op_s_cnt = np.copy(self.op_sig_model_dict[k])
            op_s_cnt = op_s_cnt + len(self.op_sig_cnts) * self.op_sig_cnts
            self.op_sig_model_dict[k] = self.norm_cnt(op_s_cnt)




class RawMaterialModel:
    def __init__(self, leaf_idx):
        self.reset()
        self.leaf_idx = leaf_idx
        self.label_map = ['not_raw', 'raw']

    def reset(self):
        # self.model = MultinomialNB()
        self.model = {'not_raw': Counter(), 'raw': Counter()}

    def save(self, fname):
        with open(fname, 'w') as f:
            pickle.dump(self.model, f, pickle.HIGHEST_PROTOCOL)

    def load(self, fname):
        with open(fname, 'r') as f:
            self.model = pickle.load(f)

    def M_step(self, actionGraphs):
        self.reset()
        str_data_dict = dict()

        str_data_dict['not_raw'] = list()
        str_data_dict['raw'] = list()

        for AG in actionGraphs:
            for action in AG.actions:
                for arg in action.ARGs:
                    if arg.sem_type == 'food':
                        for ss in arg.string_spans:
                            if ss.origin == self.leaf_idx:
                                str_data_dict['raw'].append(ss.s)
                            else:
                                str_data_dict['not_raw'].append(ss.s)

        self.model['raw'] = Counter(str_data_dict['raw'])
        self.model['not_raw'] = Counter(str_data_dict['not_raw'])

    def predict(self, S):
        prob_span_is_raw = 1
        for s in S:
            prob_span_is_raw = prob_span_is_raw * (self.model['raw'][s]/len(self.model['raw'].keys()))
                                                  # Count of string s seen as 'raw'/total 'raw' words seen

        return prob_span_is_raw






class ApparatusModel:
    def __init__(self, leaf_idx):
        self.leaf_idx = leaf_idx
        self.reset()

    def reset(self):
        self.model = None #TODO

    def save(self, fname):
        with open(fname, 'w') as f:
            pickle.dump(self.model, f, pickle.HIGHEST_PROTOCOL)

    def load(self, fname):
        with open(fname, 'r') as f:
            self.model = pickle.load(f)

    def get_all_apparatus(self, action, AG, recursive):
        aprts = []
        for arg in action.ARGs:
            if arg.sem_type == 'apparatus':
                for ss in arg.string_spans:
                    if ss.s:
                        aprts.append(ss.s)
                    elif recursive and ss.origin is not None and ss.origin != self.leaf_idx:
                        aprts.extend(self.get_all_apparatus(AG.actions[ss.origin], AG, recursive))

    def M_step(self, actionGraphs):
        for AG in actionGraphs:
            for action in AG.actions:
                op = action.op
                aprts = self.get_all_apparatus(action, AG, True)



class PartCompositeModel:

    pass

