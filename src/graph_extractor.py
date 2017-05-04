import numpy as np
import cPickle as pickle
import os

import connection_models
import graph_elements
import constants



class ActionGraphExtractor:
    def __init__(self):

        self.actionGraphs = []
        self.prior_w = 0.1

        self.leaf_idx = -1

        self.opSigModel = connection_models.OpSigModel(self.prior_w)
        self.partCompositeModel = connection_models.PartCompositeModel(self.leaf_idx)
        self.rawMaterialModel = connection_models.RawMaterialModel(self.leaf_idx)
        self.apparatusModel = connection_models.ApparatusModel(self.leaf_idx)

    def load_pretrained_models(self, load_dir):
        graph_fname = os.path.join(load_dir, constants.ACTION_GRAPH_FILE)
        with open(graph_fname, 'r') as f:
            # still load this graph to use during NLP models' M_steps
            self.actionGraphs_nlp = pickle.load(f)

        self.opSigModel.load(os.path.join(load_dir, constants.OP_SIG_MODEL_FILE))
        self.rawMaterialModel.load(os.path.join(load_dir, constants.RAW_MATERIAL_MODEL_FILE))
        self.partCompositeModel.load(os.path.join(load_dir, constants.PART_COMP_MODEL_FILE))
        self.apparatusModel.load(os.path.join(load_dir, constants.APP_MODEL_FILE))

    def save_pretrained_models(self, save_dir):
        #TODO
        pass

    def load_data(self):
        #TODO
        pass

    def evaluate_models(self, AG, verbose=False):
        log_prob = np.log(1)
        ibm_model_epsilon = 1
        num_materials = AG.get_materials_len()
        num_intrmeds = AG.get_intrmeds_len()

        opSigProb = self.opSigModel.evaluate(AG)
        log_prob += opSigProb

        if verbose:
            print 'verbSigProb ', opSigProb

        pc_prob = 0
        app_prob = 0

        raw_materials = []
        for act_i, action in enumerate(AG.actions):
            for arg_j, arg in enumerate(action.ARGs):
                for ss_k, ss in enumerate(arg.str_spans):
                    log_prob_c = np.log(1)
                    if arg.sem_type == 'material':
                        if ss.origin == self.leaf_idx:
                            raw_materials.append(ss.s)
                        else:
                            prob = self.partCompositeModel.evaluate(act_i, arg_j, ss_k, AG)
                            pc_prob += pc_prob + prob

                    elif arg.sem_type == 'location':
                        if ss.origin == self.leaf_idx:
                            # TODO: decide what to do
                            # log_prob_c = log_prob_c
                            pass
                        else:
                            prob = self.apparatusModel.evaluate(act_i, arg_j, ss_k, AG)
                            app_prob = app_prob + prob
                    log_prob = log_prob + app_prob + (ibm_model_epsilon/((num_materials+1)**num_intrmeds))*pc_prob

        raw_probs = self.rawMaterialModel.evaluate(raw_materials)
        if verbose:
            print 'rawMaterialProb ', np.sum(np.log(raw_probs))

        log_prob = log_prob + np.sum(np.log(raw_probs))
        if verbose:
            print 'full prob ', log_prob

        return log_prob





    def local_search(self, AG):
        improved = True
        max_iter = 10
        iter = 0
        p_AG = self.evaluate_models(AG)


        pass

