from __future__ import unicode_literals
import numpy as np
import cPickle as pickle
import os, sys, codecs
import logging

# My imports.
import connection_models
import graph_elements
import agex_settings
import utils


class ActionGraphExtractor:
    def __init__(self):
        self.actionGraphs = []
        self.prior_w = 0.1

        self.leaf_idx = agex_settings.LEAF_INDEX

        self.opSigModel = connection_models.OpSigModel(self.prior_w)
        self.partCompositeModel = connection_models.PartCompositeModel(self.leaf_idx)
        self.rawMaterialModel = connection_models.RawMaterialModel(self.leaf_idx)
        self.apparatusModel = connection_models.ApparatusModel(self.leaf_idx)

        self.dump_dir = 'tmp'

    def load_pretrained_models(self, load_dir):
        graph_fname = os.path.join(load_dir, agex_settings.ACTION_GRAPH_FILE)
        with open(graph_fname, 'r') as f:
            # still load this graph to use during NLP models' M_steps
            self.actionGraphs_nlp = pickle.load(f)

        self.opSigModel.load(os.path.join(load_dir, agex_settings.OP_SIG_MODEL_FILE))
        self.rawMaterialModel.load(os.path.join(load_dir, agex_settings.RAW_MATERIAL_MODEL_FILE))
        self.partCompositeModel.load(os.path.join(load_dir, agex_settings.PART_COMP_MODEL_FILE))
        self.apparatusModel.load(os.path.join(load_dir, agex_settings.APP_MODEL_FILE))

    def save_pretrained_models(self, save_dir):
        graph_fname = os.path.join(save_dir, agex_settings.ACTION_GRAPH_FILE)
        with open(graph_fname, 'r') as f:
            # still load this graph to use during NLP models' M_steps
            self.actionGraphs_nlp = pickle.load(f)

        self.opSigModel.load(os.path.join(save_dir, agex_settings.OP_SIG_MODEL_FILE))
        self.rawMaterialModel.load(os.path.join(save_dir, agex_settings.RAW_MATERIAL_MODEL_FILE))
        self.partCompositeModel.load(os.path.join(save_dir, agex_settings.PART_COMP_MODEL_FILE))
        self.apparatusModel.load(os.path.join(save_dir, agex_settings.APP_MODEL_FILE))

    def load_parsed_recipes(self, doi_file, db_name, collection_name, tar_task,
                            parsed_file_suffix):
        """
        Take an opened file with dois in it and read the parsed papers from
        disk.
        :param doi_file: file object; each line is a doi.
        :param db_name: string; says which db data came from.
        :param collection_name: string; says which collection data came from.
        :param set_suffix: string; says if this is the train/test/dev split.
        :return: None.
        """
        recipe_count = 0
        for doi_line in doi_file:
            doi_str = doi_line.strip()
            # Get paper data we care about.
            actions_li = utils.read_parsed_paper(
                doi_str, db_name=db_name, collection_name=collection_name,
                tar_task=tar_task, parsed_file_suffix=parsed_file_suffix)
            if actions_li:
                print(u'Read: {}'.format(doi_line))
                self.actionGraphs.append(graph_elements.ActionGraph(
                    parsed_actions_li=actions_li, paper_doi=doi_str))
                recipe_count += 1

        print(u'EXTRACTOR: Loaded {} recipes'.format(recipe_count))

    def evaluate_models(self, AG, verbose=False):
        log_prob = np.log(1)
        ibm_model_epsilon = 1
        num_materials = AG.get_materials_len()
        num_intrmeds = AG.get_intrmeds_len()
        opSigProb = self.opSigModel.evaluate(AG)
        log_prob += opSigProb

        if verbose:
            print(u'opSigProb: {}'.format(opSigProb))

        pc_prob = 0
        app_prob = 0

        raw_materials = []
        for act_i, action in enumerate(AG.actions):
            for arg_j, arg in enumerate(action.ARGs):
                for ss_k, ss in enumerate(arg.str_spans):
                    log_prob_c = np.log(1)
                    if arg.sem_type == agex_settings.MATERIAL_TAG:
                        # if ss.origin == self.leaf_idx:
                        raw_materials.append(ss.s)
                    elif arg.sem_type == agex_settings.INTERMEDIATE_PRODUCT_TAG:
                        prob = self.partCompositeModel.evaluate(act_i, arg_j, ss_k, AG)
                        pc_prob = pc_prob + prob

                    elif arg.sem_type == agex_settings.APPARATUS_TAG:
                        # I think that if its the leaf index then let it stay.
                        # because now that the apparatus is also seq_inited
                        # only the first one has this case and there's no place
                        # but -1 that it can connect to.
                        if ss.origin == self.leaf_idx:
                            # TODO: decide what to do
                            # log_prob_c = log_prob_c
                            pass
                        else:
                            prob = self.apparatusModel.evaluate(act_i, arg_j, ss_k, AG)
                            app_prob += prob
                    log_prob = log_prob + app_prob + np.log(ibm_model_epsilon) \
                                - num_intrmeds * np.log(num_materials+1) + pc_prob

        raw_probs = self.rawMaterialModel.evaluate(raw_materials)
        if verbose:
            print(u'rawMaterialProb: {}'.format(np.sum(raw_probs)))

        log_prob = log_prob + np.sum(raw_probs)
        if verbose:
            print(u'full prob: {}'.format(log_prob))
        return log_prob

    def get_new_sem_type(self, action):
        # get output sem_type
        sem_type_set = set()
        for arg in action.ARGs:
            sem_type_set = sem_type_set.union([arg.sem_type])

        ori_sem_type = None  # origin's semantic type
        # if there is material then it is material
        # elif there is apparatus => apparatus
        # TODO: see example that is neither
        # TODO: see if material can be referred to without making it intermed
        if 'material' in sem_type_set:
            ori_sem_type = 'intrmed'
        elif 'apparatus' in sem_type_set:
            ori_sem_type = 'appparatus'
        else:
            ori_sem_type = 'other'

        return ori_sem_type

    # AG, prev_si, cur_si
    def OP_2SWAP(self, AG, s1_idx, s2_idx):
        ss1 = AG.get_str_span(s1_idx)
        ss2 = AG.get_str_span(s2_idx)

        ori1_i = ss1.origin
        ori2_i = ss2.origin

        act1_i, arg1_j, ss1_k = s1_idx
        act2_i, arg2_j, ss2_k = s2_idx

        ss1_sem_type = AG.actions[act1_i].ARGs[arg1_j].sem_type
        ss2_sem_type = AG.actions[act2_i].ARGs[arg2_j].sem_type

        # print ori1_i, ori2_i
        # print act1_i, act2_i

        # same origin, no need to swap
        if ori1_i == ori2_i:
            # print "Same origin"
            return False

        # output to previous act is impossible
        if ori1_i >= act2_i or ori2_i >= act1_i:
            # print "Origin greater than action"
            return False

        # Don't swap if swap causes intermediate to link to -1.
        if ((ss2_sem_type is agex_settings.INTERMEDIATE_PRODUCT_TAG) and
                (ori1_i == -1)) or \
            ((ss1_sem_type is agex_settings.INTERMEDIATE_PRODUCT_TAG) and
                (ori2_i == -1)):
            return False

        arg1 = AG.actions[act1_i].ARGs[arg1_j]
        arg2 = AG.actions[act2_i].ARGs[arg2_j]

        # if arg1.sem_type != arg2.sem_type and arg1.sem_type != '' and arg2.sem_type != '':
        #     return False

        # make sure still have at least one valid DOBJ
        # => if no str_span, then cannot switch to -1
        if arg1.syn_type == 'DOBJ' and ss1.s == '' and ori2_i == self.leaf_idx \
        or arg2.syn_type == 'DOBJ' and ss2.s == '' and ori1_i == self.leaf_idx:
            # print "no str_span, cannot switch to -1"
            return False

        act1 = AG.actions[act1_i]
        act2 = AG.actions[act2_i]

        # ori_act1_out = self.get_new_sem_type(act1)
        # ori_act2_out = self.get_new_sem_type(act2)

        # swap
        ori_sem_type1 = arg1.sem_type
        ori_sem_type2 = arg2.sem_type

        if ori_sem_type2 == '':
            if ori_sem_type1 == agex_settings.MATERIAL_TAG:
                arg1.set_sem_type(agex_settings.INTERMEDIATE_PRODUCT_TAG)
            else:
                arg1.set_sem_type(ori_sem_type2)
        if ori_sem_type1 == '':
            if ori_sem_type2 == agex_settings.MATERIAL_TAG:
                arg2.set_sem_type(agex_settings.INTERMEDIATE_PRODUCT_TAG)
            else:
                arg2.set_sem_type(ori_sem_type1)

        ss1.set_origin(ori2_i)
        ss2.set_origin(ori1_i)

        act1.update_isLeaf()
        act2.update_isLeaf()
        #TODO - Update implicit argument sem type


        # # check if make other connection infeasible
        # new_act1_out = self.get_new_sem_type(act1)
        # new_act2_out = self.get_new_sem_type(act2)
        # if ori_act1_out != new_act1_out or ori_act2_out != new_act2_out:
        #     # reset swap
        #     arg1.set_sem_type(ori_sem_type1)
        #     arg2.set_sem_type(ori_sem_type2)
        #     ss1.set_origin(ori1_i)
        #     ss2.set_origin(ori2_i)
        #     act1.update_isLeaf()
        #     act2.update_isLeaf()
        #     #TODO -Update implicit argument sem type
        return True

    def local_search(self, AG):
        improved = True
        max_iter = 1
        iter = 0
        p_AG = self.evaluate_models(AG)

        num_changes = 0

        while improved and iter < max_iter:
            assert not(iter > 0 and np.isinf(p_AG))
            print(u'fishy iter: {:d}'.format(iter))
            # logging.debug('iter: %d' % iter)
            improved = False
            prev_ss_idx = []
            for act_i, act in enumerate(AG.actions):
                for arg_j, arg in enumerate(act.ARGs):
                    for ss_k, ss in enumerate(arg.str_spans):
                        cur_si = (act_i, arg_j, ss_k)
                        for prev_si in prev_ss_idx:
                            a_prev, _, _ = prev_si
                            a_cur, _, _ = cur_si
                            # ori_ap_str = AG.actions[a_prev].__str__()
                            # ori_ac_str = AG.actions[a_cur].__str__()
                            swapped = self.OP_2SWAP(AG, prev_si, cur_si)
                            if swapped:
                                new_p = self.evaluate_models(AG)

                                if new_p > p_AG:
                                    print('Change:{} ; Prob improved from {:f} '
                                          'to {:f}'.format(num_changes,
                                                           p_AG, new_p))
                                    p_AG = new_p
                                    improved = True
                                    num_changes = num_changes + 1

                                else:
                                    swapped_back = self.OP_2SWAP(AG, prev_si, cur_si)

                                    if not swapped_back:
                                        print 'error'

                                    #for debugging
                                    swapped_back_p = self.evaluate_models(AG)
                                    if swapped_back_p != p_AG:
                                       print 'error, not swapped back to the same graph'
                                       tmp = self.OP_2SWAP(AG, prev_si, cur_si)
                                    assert swapped_back_p == p_AG

                        prev_ss_idx.append(cur_si)
            iter = iter + 1

        return num_changes


    def local_search_all(self):
        # find the best graph locally
        num_changes = 0
        for i, AG in enumerate(self.actionGraphs):
            print(u'\n\nGraph {:d}: doi: {}'.format(i, AG.paper_doi))
            num_changes_graph = self.local_search(AG)
            num_changes = num_changes + num_changes_graph

            # if i % 100 == 0:
            #     tmp_save_fname = os.path.join(self.dump_dir, 'actionGraphs_%d.pkl' % i)
            #     with open(tmp_save_fname, 'w') as f:
            #         pickle.dump(self.actionGraphs, f, pickle.HIGHEST_PROTOCOL)

        return num_changes

    def M_step_all(self):
        print 'EXTRACTOR: M step all'
        AGs = self.actionGraphs
        # print self.actionGraphs
        self.opSigModel.M_step(AGs)
        self.partCompositeModel.M_step(AGs)
        self.rawMaterialModel.M_step(AGs)
        self.apparatusModel.M_step(AGs)

    def print_all_models(self):
        """
        Print all the learnt models.
        :return: None.
        """
        self.opSigModel.print_model()
        self.partCompositeModel.print_model()
        self.apparatusModel.print_model()
        self.rawMaterialModel.print_model()