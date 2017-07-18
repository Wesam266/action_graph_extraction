import os, sys
import pprint, re
import json, codecs, copy
import agex_settings


class StringSpan:
    def __init__(self, s):
        self.s = s.strip()
        self.origin = None

    def set_origin(self, i):
        self.origin = i

    def ss_to_nx(self):
        """
        Return a representation of the string span which can be used as node
        attributes in a networkx graph.
        :return: dict
        """
        span_d = {
            u'str_span': self.s,
            u'origin': self.origin,
            u'ss_id': None  # id of the string span has to be set below.
        }
        return span_d

    def __unicode__(self):
        s = self.s
        s = s + u' (' + unicode(self.origin) + u')'
        return s

    def __str__(self):
        return unicode(self).encode('utf-8')


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

    def arg_to_nx(self):
        """
        Convert the string spans in the argument to a list of dicts which can
        be used networkx node attributes.
        :return: list(dict)
        """
        spans = list()
        for ss_idx, ss in enumerate(self.str_spans):
            aug_ss = copy.deepcopy(ss.ss_to_nx())
            aug_ss[u'syn_type'] = self.syn_type
            aug_ss[u'sem_type'] = self.sem_type
            aug_ss[u'isImplicit'] = self.isImplicit
            aug_ss[u'isArg'] = True
            aug_ss[u'ss_id'] = ss_idx
            aug_ss[u'arg_id'] = None  # id of the argument has to be set below.
            spans.append(aug_ss)
        return spans

    def __unicode__(self):
        s = u'[' + self.syn_type + u', ' + self.sem_type + u'] '
        s = s + u'prep: ' + self.prep + u'; '
        s = s + u'string spans: '
        for sp in self.str_spans:
            s = s + sp.__unicode__() + u' , '
        return s

    def __str__(self):
        return unicode(self).encode('utf-8')


class Action:
    def __init__(self, parsed_action_dict):
        """

        :param parsed_action_dict:
        """
        self.ARGs = list()
        self.is_leaf = False
        self.op = parsed_action_dict[u'operation']

        materials = copy.deepcopy(parsed_action_dict[u'mat_nsubjpass'])
        materials.extend(parsed_action_dict[u'mat_prepp'])

        intermeds = copy.deepcopy(parsed_action_dict[u'int_nsubjpass'])
        intermeds.extend(parsed_action_dict[u'int_prepp'])

        apparatus = copy.deepcopy(parsed_action_dict[u'app_nsubjpass'])
        apparatus.extend(parsed_action_dict[u'app_prepp'])

        # Removing these empty strings for now because we're conditionally
        # adding them below but idk if I should do this. I added the empty
        # strings as a hack to solve a different problem so I think its okay.
        materials = [mat for mat in materials if mat != u' ']
        intermeds = [int for int in intermeds if int != u' ']
        apparatus = [app for app in apparatus if app != u' ']

        # Add an implicit argument only if there are no apparatus or intermeds.
        # Different from what the paper does.
        if len(apparatus) == 0:
            apparatus.append('')  # Implicit argument
        if len(intermeds) == 0:
            intermeds.append('')  # Implicit argument.
        # Information about DOBJ or PP needs to come from a dependency parse.
        self.ARGs.append(Argument(materials, 'DOBJ', 'material', origin=agex_settings.LEAF_INDEX))
        self.ARGs.append(Argument(apparatus, 'DOBJ', 'apparatus', origin=agex_settings.LEAF_INDEX))
        self.ARGs.append(Argument(intermeds, 'DOBJ', 'intrmed'))

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
                    if s.origin != agex_settings.LEAF_INDEX:
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

    def act_to_nx(self):
        """
        Return the action and the arguments of the action as a list of dicts.
        :return: list(dict)
        """
        act_list = list()
        act_node = {
            u'operation': self.op,
            u'act_id': None,  # Index of the action which *HAS* to be set.
            u'isArg': False  # Its an operation if its not an argument.
        }
        act_list.extend([act_node])
        for arg_idx, ARG in enumerate(self.ARGs):
            arg_dicts = copy.deepcopy(ARG.arg_to_nx())
            for arg_dict in arg_dicts:
                arg_dict[u'arg_id'] = arg_idx
            act_list.extend(arg_dicts)
        return act_list

    def __unicode__(self):
        s = u'Action {:d}: '.format(self.idx) + u' Operation: {}'.format(self.op) + u'\n'
        for arg in self.ARGs:
            s = s + arg.__unicode__() + u'\n'
        return s

    def __str__(self):
        return unicode(self).encode('utf-8')


class ActionGraph():
    def __init__(self, parsed_actions_li, paper_doi):
        """

        :param parsed_actions_li: list(dict())
        :param paper_doi: string
        """
        self.paper_doi = paper_doi
        self.actions = list()

        for parsed_action_dict in parsed_actions_li:
            self.actions.append(Action(parsed_action_dict))

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
                    count += 1
        return count

    def get_apparatus_len(self):
        count=0
        for action in self.actions:
            for arg in action.ARGs:
                if arg.sem_type == 'apparatus':
                    count+=1
        return count

    def seq_init(self):
        # This removing intermediate looks like a hack.
        for i, arg in enumerate(self.actions[0].ARGs):
            if arg.sem_type == 'intrmed':
                self.actions[0].rm_arg(i)
        self.actions[0].set_isLeaf()
        for i, act in enumerate(self.actions):
            # If not the first action
            if i > agex_settings.LEAF_INDEX+1:
                for arg in act.ARGs:
                    # Connect apparatus or intermediate to previous action.
                    # TODO: Look into whether setting apparatus sequentially
                    # does anything bad.
                    if arg.sem_type in ['intrmed', 'apparatus']:
                        for ss in arg.str_spans:
                            ss.set_origin(i-1)

    def _nx_node_id(self, act_id, arg_id, ss_id):
        """
        Given the node ids return a string to use as the node id in the networkx
        graph.
        :return: string.
        """
        if arg_id != None:
            out_str = u'act-{}_arg-{}_ss-{}'.format(act_id, arg_id, ss_id)
            return out_str
        else:
            out_str = u'act-{}'.format(act_id)
            return out_str

    def _nx_get_edges(self, node_dict):
        """
        Return edges as a list of tuples.
        :param node_dict:
        :return:
        """
        node_edges = list()
        # Edge to the origin.
        if node_dict[u'origin'] != -1:
            source = self._nx_node_id(node_dict[u'origin'], None, None)
            dest = self._nx_node_id(node_dict[u'act_id'], node_dict[u'arg_id'],
                                    node_dict[u'ss_id'])
            node_edges.extend([tuple((source, dest))])
        # Always add the edge to the current operation.
        source = self._nx_node_id(node_dict[u'act_id'], None, None)
        dest = self._nx_node_id(node_dict[u'act_id'], node_dict[u'arg_id'],
                                node_dict[u'ss_id'])
        node_edges.extend([tuple((source, dest))])
        return node_edges

    def _nx_get_nodes_edges(self):
        """
        Get a list of nodes as dicts and a list of edge tuples.
        :return:
        """
        # TODO: Try to do a better (consistent) job with populating this node
        # attr dict. - medium-priority
        nodes = list()
        # Get the nodes for each action and set order indices of the operations
        # and nominally, the arguments as well.
        for aidx, action in enumerate(self.actions):
            act_nodes = copy.deepcopy(action.act_to_nx())
            # Set order of operation nodes.
            for act_node in act_nodes:
                act_node[u'act_id'] = aidx
                if act_node[u'isArg']:
                    act_node[u'id'] = self._nx_node_id(act_node[u'act_id'],
                                                       act_node[u'arg_id'],
                                                       act_node[u'ss_id'])
                else:
                    act_node[u'id'] = self._nx_node_id(act_node[u'act_id'],
                                                       None, None)
            nodes.extend(act_nodes)
        # Build the list of edges; the node already has the edge info. Just get
        # it out. There might be nicer ways to do this.
        edges = list()
        for node in nodes:
            if node[u'isArg']:
                edges.extend(self._nx_get_edges(node))
        return nodes, edges

    def ag_to_nx(self):
        """
        Convert the AG to a networkx graph.
        :return:
        """
        import networkx as nx

        nodes, edges = self._nx_get_nodes_edges()
        nx_graph = nx.MultiGraph(name=self.paper_doi)

        # Add all nodes to graph
        for node in nodes:
            nx_graph.add_node(n=node[u'id'], attr_dict=node)

        # Add all edges to graph.
        for edge in edges:
            nx_graph.add_edge(u=edge[0], v=edge[1])

        return nx_graph

    def __unicode__(self):
        graph = u'DOI: {:s}\n'.format(self.paper_doi)
        for act in self.actions:
            graph += act.__unicode__() + u'\n'
        return graph

    def __str__(self):
        return unicode(self).encode('utf-8')


def test():
    paper_path = u'/iesl/canvas/smysore/material_science_ag/papers_data_json/' \
                 u'predsynth-annotated_papers-train/' \
                 u'10.1016-j.jpcs.2004.05.002_parsed.json'

    with codecs.open(paper_path, u'r', u'utf-8') as fp:
        paper_dict = json.load(fp, encoding=u'utf-8')
    pprint.pprint(paper_dict[u'actions'])

    actions_li = paper_dict[u'actions']
    paper_doi = paper_dict[u'doi']
    AG = ActionGraph(parsed_actions_li=actions_li, paper_doi=paper_doi)
    print(AG)
    graph = AG.ag_to_nx()
    print(graph.name)

if __name__ == '__main__':
    test()
