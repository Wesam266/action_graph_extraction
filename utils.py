from __future__ import unicode_literals
import os, sys
import codecs, json, re

import agex_settings
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)


def read_parsed_paper(doi_str, db_name, collection_name, tar_task,
                      parsed_file_suffix):
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
    # Form path to JSON file.
    temp_name = os.path.join(agex_settings.data_path,
                             u'{:s}-{:s}-{:s}'.format(db_name, collection_name,
                                                      tar_task),
                             re.sub(u'/', u'-', doi_str))
    paper_path = temp_name + '-' + parsed_file_suffix + u'.json'
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
