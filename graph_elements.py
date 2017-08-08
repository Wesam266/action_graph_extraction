from __future__ import unicode_literals
import os, sys
import pprint, re
import json, codecs, copy
import networkx as nx
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
        return unicode(self).encode(u'utf-8')


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
            aug_ss[u'sem_type'] = self.sem_type
            aug_ss[u'is_implicit'] = self.isImplicit
            aug_ss[u'is_arg'] = True
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
        return unicode(self).encode(u'utf-8')


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
        self.ARGs.append(Argument(materials, u'DOBJ', u'material', origin=agex_settings.LEAF_INDEX))
        self.ARGs.append(Argument(apparatus, u'DOBJ', u'apparatus', origin=agex_settings.LEAF_INDEX))
        self.ARGs.append(Argument(intermeds, u'DOBJ', u'intrmed'))

    def set_isLeaf(self):
        self.is_leaf = True

    def update_isLeaf(self):
        is_leaf = True
        for arg in self.ARGs:
            # If not an intermediate then check spans for their origins.
            if arg.sem_type != u'intrmed':
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
            u'is_arg': False  # Its an operation if its not an argument.
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
        return unicode(self).encode(u'utf-8')


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
                if arg.sem_type == u'material':
                    count+=1
        return count

    def get_intrmeds_len(self):
        count=0
        for action in self.actions:
            for arg in action.ARGs:
                if arg.sem_type == u'intrmed':
                    count += 1
        return count

    def get_apparatus_len(self):
        count=0
        for action in self.actions:
            for arg in action.ARGs:
                if arg.sem_type == u'apparatus':
                    count+=1
        return count

    def seq_init(self):
        # This removing intermediate looks like a hack.
        for i, arg in enumerate(self.actions[0].ARGs):
            if arg.sem_type == u'intrmed':
                self.actions[0].rm_arg(i)
        self.actions[0].set_isLeaf()
        for i, act in enumerate(self.actions):
            # If not the first action
            if i > agex_settings.LEAF_INDEX+1:
                for arg in act.ARGs:
                    # Connect apparatus or intermediate to previous action.
                    # TODO: Look into whether setting apparatus sequentially
                    # does anything bad.
                    if arg.sem_type in [u'intrmed', u'apparatus']:
                        for ss in arg.str_spans:
                            ss.set_origin(i-1)

    def _nx_node_id(self, node_dict):
        """
        Given the node ids return a string to use as the node id in the networkx
        graph.
        :return: string.
        """
        if node_dict[u'is_arg'] != True:
            out_str = u'{}_{}'.format(node_dict[u'act_id'],
                                      node_dict[u'operation'])
            return out_str
        else:
            # Ideally I should be able to check is_arg but idk if that works
            # as it should.
            if node_dict[u'str_span'] != u'':
                out_str = u'{}{}{}_{}'.format(node_dict[u'act_id'],
                                              node_dict[u'arg_id'],
                                              node_dict[u'ss_id'],
                                              node_dict[u'str_span'])
            else:
                out_str = u'{}{}{}_impl_arg'.format(node_dict[u'act_id'],
                                                    node_dict[u'arg_id'],
                                                    node_dict[u'ss_id'])
            return out_str

    def _nx_get_edges(self, act_nodes):
        """
        Return edges as a list of tuples.
        :param node_dict:
        :return:
        """
        # Build a map going from act_id to id.
        origin_id_map = dict()
        for act_supnode in act_nodes:
            for node in act_supnode:
                if node[u'is_arg'] != True:
                    origin_id_map[node[u'act_id']] = node[u'id']
        # This is a hack so i dont have to write a conditional below.
        origin_id_map[-1] = -1

        node_edges = list()
        for act_supnode in act_nodes:
            for node in act_supnode:
                if node[u'is_arg'] == False:
                    op_node = node
                # Connect up intermediates to their origins.
                if node[u'is_arg'] and node[u'sem_type'] == u'intrmed':
                    # Get the source and dest.
                    source = origin_id_map[node[u'origin']]
                    dest = node[u'id']
                    # Update the origin to be readable in the attr dict.
                    node[u'origin'] = origin_id_map[node[u'origin']]
                    node_edges.extend([tuple((source, dest,
                                              {u'is_parsed': False}))])
                # Connect up all nodes to the present operation.
                if node[u'is_arg']:
                    # Get the source and dest.
                    source = node[u'id']
                    dest = op_node[u'id']
                    # Update the link to be readable in the attr dict if its
                    # not been done yet.
                    if type(node[u'origin']) == u'int':
                        node[u'origin'] = origin_id_map[node[u'origin']]
                    node_edges.extend([tuple((source, dest,
                                              {u'is_parsed': True}))])
        return node_edges

    def _nx_get_nodes_edges(self):
        """
        Get a list of nodes as dicts and a list of edge tuples.
        :return:
        """
        # TODO: Try to do a better (consistent) job with populating this node
        # attr dict. - low-priority
        act_nodes = list()
        # Get the nodes for each action and set order indices of the operations
        # and nominally, the arguments as well.
        for aidx, action in enumerate(self.actions):
            nodes = copy.deepcopy(action.act_to_nx())
            # Set order of operation nodes.
            for node in nodes:
                node[u'act_id'] = aidx
                node[u'id'] = self._nx_node_id(node)
            act_nodes.append(nodes)
        # Build the list of edges; this is similar to how its done in the
        # annotations conversion.
        edges = self._nx_get_edges(act_nodes)
        # Flatten the list of list of nodes. (Each sublist is an action)
        act_nodes = [val for sublist in act_nodes for val in sublist]
        return act_nodes, edges

    def ag_to_nx(self):
        """
        Convert the AG to a networkx graph.
        :return:
        """
        nodes, edges = self._nx_get_nodes_edges()
        nx_graph = nx.MultiGraph(name=self.paper_doi)

        # Add all nodes to graph
        for node in nodes:
            nx_graph.add_node(n=node[u'id'], attr_dict=node)

        # Add all edges to graph.
        for edge in edges:
            nx_graph.add_edge(u=edge[0], v=edge[1], attr_dict=edge[2])

        return nx_graph

    def save_ag_as_nxgraph(self, path):
        """
        Save the action graph as a pickeled networkx graph to disk.
        :param path: string; path to directory to save graph in.
        :return: nx_graph
        """
        # TODO: This needs to be more flexible. Allow for specification of
        # either a learnt or seqinit suffix. --medium-priority.
        nx_graph = self.ag_to_nx()
        temp_name = os.path.join(path,re.sub(u'/', u'-', self.paper_doi))
        paper_nxg_path = temp_name + u'_learnt_nxg.pkl'
        nx.write_gpickle(nx_graph, paper_nxg_path)
        print(u'Wrote: {}'.format(paper_nxg_path))
        return nx_graph

    def __unicode__(self):
        graph = u'DOI: {:s}\n'.format(self.paper_doi)
        for act in self.actions:
            graph += act.__unicode__() + u'\n'
        return graph

    def __str__(self):
        return unicode(self).encode(u'utf-8')


# Some hacky test code. This could go elsewhere but idc for now.
def test_seq():
    paper_path = u'/iesl/canvas/smysore/material_science_ag/papers_data_json/' \
                 u'predsynth-annotated_papers-train/' \
                 u'10.1016-j.apcatb.2008.07.007_parsed.json'

    with codecs.open(paper_path, u'r', u'utf-8') as fp:
        paper_dict = json.load(fp, encoding=u'utf-8')

    actions_li = paper_dict[u'actions']
    paper_doi = paper_dict[u'doi']
    AG = ActionGraph(parsed_actions_li=actions_li, paper_doi=paper_doi)
    print(AG)


def test_nx():
    paper_path = u'/iesl/canvas/smysore/material_science_ag/papers_data_json/' \
                 u'predsynth-annotated_papers-train/' \
                 u'10.1016-j.apcatb.2008.07.007_parsed.json'

    with codecs.open(paper_path, u'r', u'utf-8') as fp:
        paper_dict = json.load(fp, encoding=u'utf-8')

    actions_li = paper_dict[u'actions']
    paper_doi = paper_dict[u'doi']
    AG = ActionGraph(parsed_actions_li=actions_li, paper_doi=paper_doi)
    print(AG)
    graph = AG.ag_to_nx()
    pprint.pprint(graph.nodes())
    pprint.pprint(graph.edges())


def test_nx_write():
    paper_path = u'/iesl/canvas/smysore/material_science_ag/papers_data_json/' \
                 u'predsynth-annotated_papers-train/' \
                 u'10.1016-j.apcatb.2008.07.007_parsed.json'
    write_path = u'/iesl/canvas/smysore/material_science_ag/papers_data_json/' \
                 u'predsynth-annotated_papers-example/'
    with codecs.open(paper_path, u'r', u'utf-8') as fp:
        paper_dict = json.load(fp, encoding=u'utf-8')

    actions_li = paper_dict[u'actions']
    paper_doi = paper_dict[u'doi']
    AG = ActionGraph(parsed_actions_li=actions_li, paper_doi=paper_doi)
    graph = AG.save_ag_as_nxgraph(write_path)
    pprint.pprint(graph.nodes())
    pprint.pprint(graph.edges())


if __name__ == u'__main__':
    if sys.argv[-1] == u'--test_nx':
        test_nx()
    elif sys.argv[-1] == u'--test_seq':
        test_seq()
    elif sys.argv[-1] == u'--test_nx_write':
        test_nx_write()
    else:
        print(u'Bunch of classes; dont run this stuff. :-P')
