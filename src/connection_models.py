# -*- coding: utf-8 -*-
'''
Code adapted from

'''

import numpy as np
import nltk
import pickle
from collections import Counter, defaultdict
import pprint, sys, decimal
from sklearn.naive_bayes import MultinomialNB

import utils
import constants

VERBOSE=True

class OpSigModel:
    """
    Model defines a distribution over the verb signatures for each op (verb)
    """
    def __init__(self, prior_w = 0.1):
        self.op_sig_cnts = np.array([0] * 8).astype(float)
        # The dict with the distributions over the op signatures for each
        # verb type.
        self.op_sig_model_dict = dict()
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

    def _get_opSig_idx(self, sig_set, is_leaf):
        # Create a binary number as a string to find the verb signature
        # of an action given its sig_set and if its a leaf. This is pretty
        # clever! :D Using strings and then just making them an
        # int in the end. (DOBJ,PP,is_leaf)
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
        # I think is_leaf is set elsewhere.
        is_leaf = action.is_leaf

        sig_set = set()
        # You only need to look at the args of the action for the 'type' of
        # the op signature for the op because all connections in the dest
        # subset converge on these args and that's what we care about.
        for arg in action.ARGs:
            if arg.sem_type == 'material':
                sig_set = sig_set.union([arg.syn_type])

        return self._get_opSig_idx(sig_set, is_leaf)

    def get_opSig_from_idx(self, idx):
        # Get a binary representation of idx as a string.
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
        res = float(1) / len(cnt)
        cnt[cnt == 0] = res

        s = sum(cnt)
        # assert s != 0
        if s > 0:
            prob = cnt / s
            return prob

    def M_step(self, actionGraphs):
        """
        Update parameters of model using current set of connections.
        :param actionGraphs:
        :return:
        """
        self.reset()

        # Form counts of op signatures for each op.
        for AG in actionGraphs:
            for action in AG.actions:
                op = action.op
                idx = self.get_opSig_idx(action)
                # Looks like setting up for some kind of
                # smoothing to me but I'm not sure of this yet.
                # This might be some kind of global prior on the distribution
                # for each verb. Thats what hschang said.
                self.op_sig_cnts[idx] += 1

                # If op not in model dict add it.
                if op not in self.op_sig_model_dict:
                    self.op_sig_model_dict[op] = np.array([0] * 8).astype(float)
                # Increment the count for the verb signature for the op.
                self.op_sig_model_dict[op][idx] += 1

        # normalize probs
        self.op_sig_cnts = self.norm_cnt(self.op_sig_cnts)
        for op in self.op_sig_model_dict.keys():
            # Copy so you aren't touching the original array.
            op_s_cnt = np.copy(self.op_sig_model_dict[op])
            # This is where that global prior is getting added but im
            # not sure of this.
            op_s_cnt = op_s_cnt + len(self.op_sig_cnts) * self.op_sig_cnts
            self.op_sig_model_dict[op] = self.norm_cnt(op_s_cnt)
        if VERBOSE:
            print 'Opsig model: ', pprint.pprint(self.op_sig_model_dict)


    def evaluate(self, actionGraph):
        """
        Evaluate probability of current action graph under the existing model.
        :param actionGraph:
        :return:
        """
        log_prob = np.log(1)
        for action in actionGraph.actions:
            op = action.op
            vb_idx = self.get_opSig_idx(action)
            if op in self.op_sig_model_dict:
                # I guess this is okay but why exactly multiply with that
                # prior_w here?
                log_prob += np.log(self.op_sig_model_dict[op][vb_idx]) + \
                            self.prior_w * np.log(self.op_sig_cnts[vb_idx])
            else:
                # Why not multiply with the prior here too?
                log_prob += np.log(self.op_sig_cnts[vb_idx])
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
        print 'Raw material model: ', pprint.pprint(dict(self.model))

    def evaluate(self, S):
        prob_span_is_raw = 1
        for s in S:
            prob_span_is_raw = prob_span_is_raw * (self.model['raw'][s]/len(self.model['raw'].keys()))
                                                  # Count of string s seen as 'raw'/total 'raw' words seen

        if prob_span_is_raw > 0:
            return np.log(prob_span_is_raw)
        else:
            return np.log(0.000000001)


# I think that the general implementation below here is correct but that the
# interpretation in his comment and the model that this prints might be
# incorrect.
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
            print 'Apparatus Model: ', pprint.pprint(dict(self.model))




    def evaluate(self, action_i, arg_j, ss_k, AG):

        assert AG.actions[action_i].ARGs[arg_j].sem_type == 'apparatus', "ERROR: Not Apparatus"
        ss = AG.actions[action_i].ARGs[arg_j].str_spans[ss_k]
        op = AG.actions[action_i].op
        ori_act_i = ss.origin
        assert ori_act_i != self.leaf_idx
        aprts = self.get_all_apparatus(AG.actions[ori_act_i], AG, False)
        # If there is a string (i.e its not an implicite argument) then its
        # deterministic and either 1 or 0.
        if ss.s:
            found = utils.substr_match(ss.s, aprts)
            if found:
                return np.log(1.0)
            else:
                # Paper says to use 1 or 0 but we cant use a zero because
                # of using the log? We could, provided all the other code
                # supported that too. But im not sure yet.
                return np.log(0.000000001) #TODO: Can we change this?

        #TODO: Verify that the math is correct
        # If there the span is empty (its an implicit argument) then
        # use the prob that the location arg at the origin verb occuring
        # with that origin verb.
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
    """
    Model assigns a string span a probability when its an intermediate material.
    Model says what materials are likely to have originated from what other
    materials.
    """
    def __init__(self, leaf_idx):
        # I guess there's no reason why a Counter must be used here.
        # You could just use a defaultdict(int). Because I guess we're
        # not really using the Counters abilities.
        self.model = defaultdict(Counter)
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

    def get_action_intermeds(self, action):
        all_intrmeds = []
        for arg in action.ARGs:
            if arg.sem_type == 'intrmed':
                for ss in arg.str_spans:
                    all_intrmeds.append(ss.s)
        return all_intrmeds


    def M_step(self, actionGraphs):
        self.reset()
        for AG in actionGraphs:
            for action in AG.actions:
                intrmed_prods = self.get_action_intermeds(action)
                # mtrls now is a list of lists with each sublist consisting
                # of materials from all previous actions which have a directed
                # path to the current string span.
                mtrls = list()
                for arg in action.ARGs:
                    if arg.sem_type is 'intrmed':
                        for ss in arg.str_spans:
                            # Get all materials from all the actions with a
                            # directed path to the current string span.
                            ori_mtrls = self.get_all_materials(
                                AG.actions[ss.origin], AG, True)
                            mtrls.append(ori_mtrls)
                # Ideally the length of intermed_prods and mtrls should
                # be the same.
                assert len(intrmed_prods) == len(mtrls)
                for each_interm, each_ori_mats in zip(intrmed_prods, mtrls):
                    for ori_mat in each_ori_mats:
                        if each_interm:
                            self.model[each_interm][ori_mat] += 1
                        else:
                            self.model['IMPLCT_ARG'][ori_mat] += 1

        # Using this to smooth the conditional distr in the evaluate
        # function below. This can be computed here since otherwise
        # it gets computed multiple times un-necessarily in the
        # evaluate function.
        self.val_sum = 0
        for intermed in self.model.keys():
            self.val_sum += len(self.model[intermed].values())

        if VERBOSE:
            print 'Part Composite model: ', pprint.pprint(dict(self.model))


    def evaluate(self, action_i, arg_j, ss_k, AG):
        """
        I think this evaluates one probability term in the ibm-model-1 based
        on the counts formed in the M_step.
        One t(f_i|e_{a_i}) term. So for a prob over all string spans this
        needs to get called m (number of string spans in the arg) times.
        """
        assert AG.actions[action_i].ARGs[arg_j].sem_type == 'intrmed',\
            "ERROR: Not Intermediate product"
        ss = AG.actions[action_i].ARGs[arg_j].str_spans[ss_k]
        ori_act_i = ss.origin
        # print action_i, arg_j, ss_k, ss.origin
        # print ss.s
        # assert ori_act_i != self.leaf_idx

        # Get materials from all prior actions with a directed path to ss
        span_mtrls = self.get_all_materials(AG.actions[ss.origin], AG, True)
        cnt = 0
        for m in span_mtrls:
            if ss.s:
                cnt += self.model[ss.s][m]
            else:
                cnt += self.model['IMPLCT_ARG'][m] #skc: could be incorrect

        # Conditional prob part; present given previous; Laplace smoothed.
        alpha = 0.1
        cp = (cnt + alpha)/(sum(self.model[ss.s].values()) + alpha*self.val_sum)

        # Uniform prob factor. 1/[1+(number of spans in current action)]
        # This is slightly incorrect for now, since you want the number
        # of source string span groups here. But idk how to get that.
        up = 1.0/(len(AG.actions[action_i].ARGs[arg_j].str_spans)+1)
        # Spans contribution to entire prob.
        res = np.log(up) + np.log(cp)

        return res
