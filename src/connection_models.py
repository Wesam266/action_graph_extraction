# -*- coding: utf-8 -*-
'''
Code adapted from

'''

import numpy as np
import nltk
import pickle
from collections import Counter, defaultdict
from sklearn.naive_bayes import MultinomialNB

import utils
import constants

VERBOSE=True

class OpSigModel:
    def __init__(self, prior_w = 0.1):
        self.reset()
        self.prior_w = prior_w
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
                self.op_sig_model_dict[op][idx] += 1

        # normalize probs
        self.op_sig_cnts = self.norm_cnt(self.op_sig_cnts)
        for k in self.op_sig_model_dict.keys():
            op_s_cnt = np.copy(self.op_sig_model_dict[k])
            op_s_cnt = op_s_cnt + len(self.op_sig_cnts) * self.op_sig_cnts
            self.op_sig_model_dict[k] = self.norm_cnt(op_s_cnt)
        if VERBOSE:
            print 'Opsig model: ', self.op_sig_model_dict


    def evaluate(self, actionGraph):
        log_prob = np.log(1)
        for action in actionGraph.actions:
            op = action.op
            vb_idx = self.get_opSig_idx(action)
            if op in self.op_sig_model_dict.keys():
                log_prob = log_prob + self.prior_w * np.log(self.op_sig_cnts[vb_idx])
                log_prob = log_prob + np.log(self.op_sig_model_dict[op][vb_idx])
            else:
                log_prob = log_prob + np.log(self.op_sig_cnts[vb_idx])
        return log_prob






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
                    if arg.sem_type == 'material':
                        for ss in arg.str_spans:
                            # if ss.origin == self.leaf_idx:
                            str_data_dict['raw'].append(ss.s)
                            # else:
                    elif arg.sem_type == constants.INTERMEDIATE_PRODUCT_TAG:
                        for ss in arg.str_spans:
                            str_data_dict['not_raw'].append(ss.s)

        self.model['raw'] = Counter(str_data_dict['raw'])
        self.model['not_raw'] = Counter(str_data_dict['not_raw'])
        print 'Raw material model: ', self.model

    def evaluate(self, S):
        prob_span_is_raw = 1
        for s in S:
            prob_span_is_raw = prob_span_is_raw * (self.model['raw'][s]/len(self.model['raw'].keys()))
                                                  # Count of string s seen as 'raw'/total 'raw' words seen

        if prob_span_is_raw > 0:
            return np.log(prob_span_is_raw)
        else:
            return np.log(0.000000001)






class ApparatusModel:
    def __init__(self, leaf_idx):
        self.leaf_idx = leaf_idx
        self.reset()

    def reset(self):
        self.model = defaultdict(Counter)
        # models how likely it is that an action v_i occurs in the location corresponding to a origin verb
        # P(loc(origin(s_{ij}^k,C))| v_i)

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
                for ss in arg.str_spans:
                    if ss.s:
                        aprts.append(ss.s)
                    elif recursive and ss.origin is not None and ss.origin != self.leaf_idx:
                        aprts.extend(self.get_all_apparatus(AG.actions[ss.origin], AG, recursive))

        return aprts

    def M_step(self, actionGraphs):
        self.reset()

        for AG in actionGraphs:
            for action in AG.actions:
                op = action.op
                aprts = self.get_all_apparatus(action, AG, False)
                for a in aprts:
                    self.model[op][a] += 1
        self.val_sum = 0
        for i in self.model.keys():
            self.val_sum += len(self.model[i].values())
        if VERBOSE:
            print 'Apparatus Model: ', self.model




    def evaluate(self, action_i, arg_j, ss_k, AG):

        assert AG.actions[action_i].ARGs[arg_j].sem_type == 'apparatus', "ERROR: Not Apparatus"
        ss = AG.actions[action_i].ARGs[arg_j].str_spans[ss_k]
        op = AG.actions[action_i].op
        ori_act_i = ss.origin
        assert ori_act_i != self.leaf_idx
        aprts = self.get_all_apparatus(AG.actions[ori_act_i], AG, False)
        if ss.s:
            found = utils.substr_match(ss.s, aprts)
            if found:
                return np.log(1.0)
            else:
                return np.log(0.000000001) #TODO: Can we change this?

        #TODO: Verify that the math is correct
        else:
            # aprts = self.get_all_apparatus(AG.actions[ori_act_i], AG, False)
            max_val = 0
            max_aprts = ''
            for a in aprts:
                if self.model[op][a] > max_val:
                    max_val = self.model[op][a]
                    max_aprts = a
            op_sum = sum(self.model[op].values())
            alpha = 1.0
            return np.log((alpha*(max_val+1))/(op_sum + alpha*self.val_sum))





class PartCompositeModel:
    def __init__(self, leaf_idx):
        self.reset()
        self.leaf_idx = leaf_idx
        pass

    def reset(self):
        self.model = defaultdict(Counter)

    def save(self, fname):
        with open(fname, 'w') as f:
            pickle.dump(self.model, f, pickle.HIGHEST_PROTOCOL)

    def load(self, fname):
        with open(fname, 'r') as f:
            self.model = pickle.load(f)



    def get_all_materials(self, action, AG, recursive):
        all_materials = []

        for arg in action.ARGs:
            if arg.sem_type == 'material' or arg.sem_type == 'intrmed':
                for ss in arg.str_spans:
                    all_materials.append(ss.s)
                    if recursive and ss.origin != self.leaf_idx:
                        all_materials.extend(self.get_all_materials(AG.actions[ss.origin], AG, recursive))

        return all_materials

    def get_all_intermeds(self, action, AG, recursive):
        all_intrmeds = []
        for arg in action.ARGs:
            if arg.sem_type == 'intrmed':
                for ss in arg.str_spans:
                    all_intrmeds.append(ss.s)
                    if recursive and ss.origin != self.leaf_idx:
                        all_intrmeds.extend(self.get_all_materials(AG.actions[ss.origin], AG, recursive))
        return all_intrmeds


    def M_step(self, actionGraphs):
        self.reset()
        for AG in actionGraphs:
            for action in AG.actions:

                op = action.op
                mtrls = self.get_all_materials(action, AG, False) #has to be changed to true
                intrmed_prods = self.get_all_intermeds(action, AG, False)

                for i in intrmed_prods:
                    for m in mtrls:
                        if i:
                            self.model[i][m] += 1
                        else:
                            self.model['IMPLCT_ARG'][m] += 1
        self.val_sum = 0
        for i in self.model.keys():
            self.val_sum += len(self.model[i].values())

        if VERBOSE:
            print 'Part Composite model: ', self.model


    def evaluate(self, action_i, arg_j, ss_k, AG):
        assert AG.actions[action_i].ARGs[arg_j].sem_type == 'intrmed', "ERROR: Not Intermediate product"
        ss = AG.actions[action_i].ARGs[arg_j].str_spans[ss_k]
        op = AG.actions[action_i].op
        ori_act_i = ss.origin
        # print action_i, arg_j, ss_k, ss.origin
        # print ss.s
        assert ori_act_i != self.leaf_idx
        mtrls = self.get_all_materials(AG.actions[action_i], AG, False)
        cnt = 0
        for m in mtrls:
            if ss.s:
                cnt += self.model[ss.s][m]
            else:
                cnt += self.model['IMPLCT_ARG'][m] #could be incorrect
        alpha = 1.0

        return np.log((alpha*(cnt+1))/(sum(self.model[ss.s].values()) + alpha*self.val_sum))
