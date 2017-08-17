from __future__ import unicode_literals
import os, sys, argparse
import codecs, json, pprint, pickle
import re
import networkx as nx

import agex_settings, connection_models
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)


def read_parsed_paper(paper_path):
    """
    Read paper asked for and return the list of actions.
    :param doi_str:
    :param db_name:
    :param collection_name:
    :param set_suffix:
    :param parsed_file_suffix:
    :return:
    """
    # TODO: Decide if there's value in reading from a serialized file. -med-pri.
    # TODO: Include some error checks for if read fails. -low-priority.
    with codecs.open(paper_path, u'r', u'utf-8') as fp:
        parsed_paper_dict = json.load(fp, encoding=u'utf-8')

    actions_li = parsed_paper_dict[u'actions']
    return actions_li


def substr_match(str_span, locs):
    for l in locs:
        match = l.find(str_span)
        if match >= 0:
            return True

    return False