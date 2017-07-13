import sys
import pprint
import constants


class StringSpan:
    def __init__(self, s):
        self.s = s
        self.origin = None

    def set_origin(self,i):
        self.origin = i

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
            self.str_spans[-1].set_origin(origin)  # Clever.

    # def get_str_in_span(self, k):
    #     return self.str_spans[k].get_str()

    # Not sure why this is there. Whats the index associated with an argument?.
    def set_idx(self, j):
        self.idx = j

    def set_sem_type(self, sem_type):
        self.sem_type = sem_type

    def get_str(self):
        s = ''
        s = s + self.prep
        for ss in self.str_spans:
            s = s + ' ' + ss.s
        return s.strip()

    def __str__(self):
        s = '[' + self.syn_type + ', ' + self.sem_type + '] '
        s = s + 'prep: ' + self.prep + '; '
        s = s + 'string spans: '
        for sp in self.str_spans:
            s = s + sp.__str__() + ' , '
        return s

    def __repr__(self):
        return self.__str__()


class Action:
    def __init__(self, sentence_annotated):
        """

        :param sentence_annotated: A list of lists. Each sublist is one token.
        """
        self.ARGs = list()
        self.is_leaf = False
        self.op = ''
        self.omitted = list()
        materials = list()
        apparatus = list()
        intermeds = list()
        material_pos_tags = list()
        apparatus_pos_tags = list()
        text = ''
        prep = ''
        temp_ops = list()
        for tok_annotation in sentence_annotated:
            text = text + ' ' + tok_annotation[0]
            # Don't assume that theres a single operation. Read them all in
            # and then pick.
            if tok_annotation[1] == 'B-operation':
                temp_ops.append(tok_annotation[0])
            elif tok_annotation[1] == 'I-operation':
                temp_ops[-1] = temp_ops[-1] + ' ' + tok_annotation[0]
            elif tok_annotation[1] == 'B-material':
                materials.append(tok_annotation[0])
                material_pos_tags.append(tok_annotation[6])
            elif tok_annotation[1] == 'I-material':
                materials[-1] = materials[-1] + ' ' + tok_annotation[0]
            elif tok_annotation[1] == 'B-synth_aprt':
                apparatus.append(tok_annotation[0])
                apparatus_pos_tags.append(tok_annotation[6])
            elif tok_annotation[1] == 'I-synth_aprt':
                apparatus[-1] = apparatus[-1] + ' ' + tok_annotation[0]
            elif tok_annotation[1] == 'B-intrmed':
                intermeds.append(tok_annotation[0])
            elif tok_annotation[1] == 'I-intrmed':
                intermeds[-1] = intermeds[-1] + ' ' + tok_annotation[0]
            # The prepositional phrases are all ignored for now.
            # elif tok_annotation[10] == 'case':
            #     prep += tok_annotation[0]
            # elif tok_annotation[10] == 'det' and prep:
            #     prep += tok_annotation[0]
            # print prep
            # prep = ''

        # Pick the first op among multiple ops
        self.op = temp_ops[0]
        # TODO: Get rid of this if statement and the ommited thing because
        # you dont really need it. --low priority.
        if self.op:
            # print self.op
            # if len(materials) == 0:
            #     materials.append('') #Implicit argument
            if len(apparatus) == 0:
                apparatus.append('') #Implicit argument
            if len(intermeds) == 0:
                # TODO Has to be removed for the first operation.
                intermeds.append('') #Implicit argument.
            # Information about DOBJ or PP needs to come from a
            # dependency parse.
            self.ARGs.append(Argument(materials, 'DOBJ', 'material', origin=constants.LEAF_INDEX))
            self.ARGs.append(Argument(apparatus, 'DOBJ', 'apparatus', origin=constants.LEAF_INDEX))
            self.ARGs.append(Argument(intermeds, 'DOBJ', 'intrmed'))
        else:
            self.omitted.append(text)

    def set_isLeaf(self):
        self.is_leaf = True

    def update_isLeaf(self):
        is_leaf = True
        for arg in self.ARGs:
            # If not an intermediate then check spans for their origins.
            if arg.sem_type != 'intrmed':
                # A single non leaf origin span makes is_leaf = False.
                if not is_leaf:
                    break
                for s in arg.str_spans:
                    if s.origin != constants.LEAF_INDEX:
                        is_leaf = False
                        break
            # If the arg is a intermediate set is_leaf to False.
            else:
                is_leaf = False
        self.is_leaf = is_leaf

    # Not sure why this is there. Whats the index associated with an action?.
    def set_idx(self, i):
        self.idx = i

    def set_arg_indices(self):
        for j in range(len(self.ARGs)):
            self.ARGs[j].set_idx(j)

    # def get_span_in_arg(self, j, k):
    #     return self.ARGs[j].get_str_in_span(k)
    #     pass

    def rm_arg(self, i):
        del self.ARGs[i]

    def __str__(self):
        s = 'Action {:d}: '.format(self.idx) + ' Operation: {}'.format(self.op) + '\n'
        for arg in self.ARGs:
            s = s + arg.__str__() + '\n'
        return s

    def __repr__(self):
        return self.__str__()


class ActionGraph():
    def __init__(self, recipe_annotated):
        """

        :param recipe_annotated: A list of lists with each sublist being an
            annotated token. Sublist with one '' element delimits sentences.
        """
        self.actions = []
        sentence = []
        sentence_ann = []
        for tok_annotation in recipe_annotated:
            # If you get to the end of a sentence initialize an action.
            if tok_annotation[0] == '' and len(tok_annotation) == 1:
                # Only add the sentence as an action if it has a op in it.
                if ('B-operation' in sentence_ann) or \
                    ('I-operation' in sentence_ann):
                    self.actions.append(Action(sentence))
                # Empty the sentence tokens and annotations irrespective!
                sentence = []
                sentence_ann = []
            # Keep appending till you get to the end of a sentence.
            else:
                sentence.append(tok_annotation)
                sentence_ann.append(tok_annotation[1])

        for i, act in enumerate(self.actions):
            act.set_idx(i)

        self.seq_init()

    def set_action_indices(self):
        for i in range(len(self.actions)):
            self.actions[i].set_idx(i)

    # def get_spans_in_action(self, i, j, k):
    #     return self.actions[i].get_span_in_arg(j, k)

    def get_str_span(self, ss_idx):
        act_i, arg_j, ss_k = ss_idx
        if ss_k < 0:
            return None
        else:
            return self.actions[act_i].ARGs[arg_j].str_spans[ss_k]

    def get_materials_len(self):
        count=0
        for action in self.actions:
            for arg in action.ARGs:
                if arg.sem_type == 'material':
                    count+=1
        return count

    def get_intrmeds_len(self):
        count=0
        for action in self.actions:
            for arg in action.ARGs:
                if arg.sem_type == 'intrmed':
                    count+=1
        return count

    def get_apparatus_len(self):
        count=0
        for action in self.actions:
            for arg in action.ARGs:
                if arg.sem_type == 'apparatus':
                    count+=1
        return count

    def seq_init(self):
        self.actions[0].set_isLeaf()
        # This removing intermediate looks like a hack.
        for i, arg in enumerate(self.actions[0].ARGs):
            if arg.sem_type == 'intrmed':
                self.actions[0].rm_arg(i)
        for i, act in enumerate(self.actions):
            # If not the first action
            if i > constants.LEAF_INDEX+1:
                for arg in act.ARGs:
                    # Connect apparatus or intermediate to previous action.
                    # TODO: Look into whether setting apparatus sequentially
                    # does anything bad.
                    if arg.sem_type in ['intrmed', 'apparatus']:
                        for ss in arg.str_spans:
                            ss.set_origin(i-1)

    def __str__(self):
        graph = ''
        for act in self.actions:
            graph += act.__str__() + '\n'
        return graph


def test():
    TRAIN_FILE = '../data/exp2.txt'
    annot = []
    for line in open(TRAIN_FILE):
        split = line.strip().split("\t")
        annot.append(split)

    print ActionGraph(annot)

if __name__ == '__main__':
    test()
