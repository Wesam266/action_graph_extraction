import numpy as np
import nltk
import re
from connection_models import OpSigModel

token_classes = ['null', 'amt_unit', 'amt_misc', 'cnd_unit', 'cnd_misc', 'material', 'target', 'operation',
                 'descriptor', 'prop_unit', 'prop_type', 'synth_aprt', 'char_aprt', 'brand', 'intrmed',
                 'number', 'meta', 'ref', 'prop_misc']


class StringSpan:
    def __init__(self, s):
        self.s = s
        self.origin = None

    def set_origin(self,i):
        self.origin = i

    def get_str(self):
        return self.s

    def __str__(self):
        s = self.s
        s = s + ' (' + str(self.origin) + ')'
        return s

    def __repr__(self):
        return self.__str__()


class Argument:
    def __init__(self, text, syn_type, sem_type, prep='', origin=-1):
        self.syn_type = syn_type
        self.sem_type = sem_type
        self.str_spans = []
        self.isImplicit = True
        self.prep = prep
        # strs = filter(None, text) #To remove empty string spans
        for s in text:
            self.str_spans.append(StringSpan(s))
            self.str_spans[-1].set_origin(origin)

    def get_str_in_span(self, k):
        return self.str_spans[k].get_str()

    def set_idx(self, j):
        self.idx = j

    def __str__(self):
        s = '[' + self.syn_type + ', ' + self.sem_type + '] '
        s = s + 'prep: ' + self.prep + '; '
        s = s + 'string spans: '
        for sp in self.str_spans:
            s = s + sp.__str__() + ' , '
        return s

    def get_str(self):
        s = ''
        s = s + self.prep
        for ss in self.str_spans:
            s = s + ' ' + ss.s
        return s.strip()

    def __repr__(self):
        return self.__str__()





class Action():
    def __init__(self, sentence_annotated):
        self.ARGs = []
        self.is_leaf = False
        self.origin = -1
        self.op = ''
        self.omitted = []
        materials = []
        apparatus = []
        intermeds = []
        material_pos_tags = []
        apparatus_pos_tags = []
        text = ''
        prep = ''
        for annotation in sentence_annotated:
            text = text + ' ' + annotation[0]
            if annotation[1] == 'B-operation':
                self.op = annotation[0]
            elif annotation[1] == 'I-operation':
                self.op = self.op + ' ' + annotation[0]
            elif annotation[1] == 'B-material':
                materials.append(annotation[0])
                material_pos_tags.append(annotation[5])
            elif annotation[1] == 'I-material':
                materials[-1] = materials[-1] + ' ' + annotation[0]
            elif annotation[1] == 'B-synth_aprt':
                apparatus.append(annotation[0])
                apparatus_pos_tags.append(annotation[5])
            elif annotation[1] == 'I-synth_aprt':
                apparatus[-1] = apparatus[-1] + ' ' + annotation[0]
            elif annotation[1] == 'B-intrmed':
                intermeds.append(annotation[0])
            elif annotation[1] == 'I-intrmed':
                intermeds[-1] = intermeds[-1] + ' ' + annotation[0]
            # elif annotation[9] == 'case':
            #     prep += annotation[0]
            # elif annotation[9] == 'det' and prep:
            #     prep += annotation[9]
            # print prep
            # prep = ''



        # print intermeds


        if not self.op:
            if len(materials) == 0:
                materials.append('') #Implicit argument
            if len(apparatus) == 0:
                apparatus.append('') #Implicist argument

            self.ARGs.append(Argument(materials, 'DOBJ', 'material', origin=0 ))
            self.ARGs.append(Argument(apparatus, 'DOBJ', 'apparatus', origin=0))
            self.ARGs.append(Argument(intermeds, 'DOBJ', 'intrmed'))
        else:
            self.omitted.append(text)



    def set_isLeaf(self):
        self.is_leaf = True
        self.origin = 0

    def set_idx(self, i):
        self.idx = i


    def set_arg_indices(self):
        for j in range(len(self.ARGs)):
            self.ARGs[j].set_idx(j)


    def get_span_in_arg(self, j, k):
        return self.ARGs[j].get_str_in_span(k)
        pass

class ActionGraph():
    def __init__(self, recipes_annotated):
        self.actions = []
        sentence = []
        for annotation in recipes_annotated:
            if annotation[0] == '' and len(annotation) == 1:
                self.actions.append(Action(sentence))
                sentence = []
            else:
                sentence.append((annotation))

        self.actions[0].set_isLeaf()
        opsig = OpSigModel()
        # for a in self.actions:
        #     print opsig.get_opSig_idx(a)


    def set_action_indices(self):
        for i in range(len(self.actions)):
            self.actions[i].set_idx(i)

    def get_spans_in_action(self, i, j, k):
        return self.actions[i].get_span_in_arg(j, k)










def test():
    TRAIN_FILE = '../data/new_crf_features_train.txt'

    annot = []
    for line in open(TRAIN_FILE):
        split = line.strip().split("\t")
        annot.append(split)

    ActionGraph(annot)

if __name__ == '__main__':
    test()














