"""
Make the MIT annotated graph into the same format as the one our
Action graph extraction code outputs.
"""
import os, sys
import pickle, codecs, pprint
import graph_elements as GE
import constants

# Directory to read the pickle files of the dictionaries which contain
# annotations.
data_path = '../data'


def read_dict(fname):
    """
    Reads the dictionary structure which comes from the MIT annotations.
    :param fname: name of the file which should get appended to data_path.
    :return: The dictionary with annotations.
    """
    with open(os.path.join(data_path, fname), 'r') as out_pd:
        ret_dict = pickle.load(out_pd)
    return ret_dict


def get_origin(material_id, connections):
    """
    Given the material_id and all connections, get the materials origin.
    :param material_id: string identifying the material.
    :param connections: list of dicts where each dict is a connection.
    :return: string identifying the origin action (operation).
    """
    for connection in connections:
        # If the id is the destination return the source.
        if material_id == connection[u'id2']:
            return connection[u'id1']


def get_op_args(operation, connections, entities):
    """
    Given an operation; all connections and all entities get the operations
    arguments. Locations and materials/intermediates.
    :param operation: dict representing the operation.
    :param connections: list of dicts where each dict is a connection.
    :param entities: list of dicts where each dict is a entity.
    :return:    apparatuses: list of strings saying the apparatus for the op.
                args: list of dicts saying which material arguments for the op.
    """
    op_id = operation[u'_id']
    # Get all the apparatus if any.
    apparatuses = list()
    for apparatus in operation['apparatuses']:
        apparatuses.append(apparatus[u'raw_text'][u'raw_texts'])

    # For all connections get the ids of nodes which have the op_id as their
    # destination.
    op_arg_ids = list()
    # Which entities have this op as their origin.
    dest_arg_ids = list()
    for conn in connections:
        if conn[u'id2'] == op_id:
            op_arg_ids.append(conn[u'id1'])
        elif conn[u'id1'] == op_id:
            dest_arg_ids.append(conn[u'id2'])

    # Get the argument strings given their ids.
    args = list()
    for arg_id in op_arg_ids:
        for entity in entities:
            arg = dict()
            if entity[u'_id'] == arg_id:
                arg[u'_id'] = entity[u'_id']
                arg[u'raw_texts'] = entity[u'raw_texts']
                args.append(arg)

    return apparatuses, args


def connect_actions(actions_li, connections):
    """
    Given the actions and the connections. Connect up all the actions (resolve
    all references).
    :param actions_li: List of dicts which represent the actions.
    :param connections: List of dicts which represent all the connections.
    :return: None. Modifies the actions in actions_li in place.
    """
    for action in actions_li:
        for material in action['materials']:
            # If an intermediate get the origin and an implicit argument.
            # (ideally an implicit arg should also have a token label; asking
            # Edward for this)
            if material[u'raw_texts'] == []:
                material[u'origin'] = get_origin(material[u'_id'], connections)
            # If an intermediate get the origin and not an implicite argument
            # (for some reason if there are many tokens they each have a label
            # assigned although they will be the same. Hence the set.)
            elif list(set(material[u'raw_texts'][0][u'token_labels']))[0] == \
                constants.token_class_dict['intrmed']:
                material[u'origin'] = get_origin(material[u'_id'], connections)
            else:
                material[u'origin'] = u'-1'


def convert_graph_formats(fname):
    """
    Convert between formats of the graphs: dishwasher annotations -> our format.
    :param fname: pickle file containing the annotations.
    :return: ideally something meaningful. Nothing now.
    """
    # TODO: Make GE constructors take the result of this and make our graph.

    # TODO: Go over the connections and put all the actions in a topological
    # TODO: sort. Also replace the string references with integers as in ours.

    # TODO: Make the apparatus also have the same dict format as the entities.

    # The dictionary which has the annotations as they are on dishwasher.
    paper_ann = read_dict(fname)

    # Get things you care about from the read in dict.
    operations = paper_ann['operations']
    connections = paper_ann['connections']
    entities = paper_ann['entities']

    # Go over the operations and get each operations input arguments.
    actions_li = list()
    for operation in operations:
        temp_dict = dict()
        temp_dict['op'] = {u'_id': operation[u'_id'],
                           u'raw_texts': operation[u'raw_texts']}
        temp_dict['apparatus'], temp_dict['materials'] = get_op_args(
            operation, connections, entities)
        actions_li.append(temp_dict)

    # Go over each action (op and its arguments) and connect up the between
    # action edges.
    connect_actions(actions_li, connections)

    pprint.pprint(actions_li)

if __name__ == '__main__':
    convert_graph_formats(fname='paper_dict.pd')