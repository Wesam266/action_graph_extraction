from __future__ import unicode_literals
import os, sys
import codecs, argparse

import graph_extractor


def run_extractors(doi_fname, db_name, collection_name, tar_task,
                   parsed_file_suffix, em_iters):

    AGE = graph_extractor.ActionGraphExtractor()
    # Load training data and initialize connections sequentially.
    with codecs.open(doi_fname, 'r') as doi_file_fh:
        AGE.load_parsed_recipes(doi_file=doi_file_fh, db_name=db_name,
                                collection_name=collection_name,
                                tar_task=tar_task,
                                parsed_file_suffix=parsed_file_suffix)

    # Do em_iters iterations of E and M.
    for i in range(em_iters):
        print(u'\n\n\nEM iteration:{}'.format(i))
        AGE.M_step_all()
        num_changes = AGE.local_search_all()
        print(u'num_changes: {:d}'.format(num_changes))
        if num_changes < 2:
            break

    # Print models and graphs learnt.
    AGE.print_all_models()
    for i, AG in enumerate(AGE.actionGraphs):
        print(u'\n\n\nAction graph {}'.format(i))
        # Not using this unicode thing here gives me some very annoying errors.
        print(unicode(AG))


def main():
    """
    Parse all command line args and call appropriate functions.
    :return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(u'-d', u'--db_name',
                        choices=[u'predsynth'],
                        default=u'predsynth',
                        help=u'Name of database data came from.')
    parser.add_argument(u'-c', u'--collection_name',
                        choices=[u'annotated_papers', u'papers'],
                        default=u'annotated_papers',
                        help=u'Mongo collection data came from.')
    parser.add_argument(u'-t', u'--tar_task',
                        choices=['age'],
                        default='age',
                        help=u'Target task of data; says which directory to '
                             u'read')
    parser.add_argument(u'-s', u'--parsed_file_suffix',
                        required=True,
                        choices=['deps_heu_parsed'],
                        help=u'Suffix for the json files which get written. '
                             u'This is to allow data parsed in different ways '
                             u'to remain side by side')
    parser.add_argument(u'-f', u'--doi_file',
                        required=True, help=u'Path to text file with DOIs.')
    parser.add_argument(u'--em_iters', type=int, required=True,
                        help=u'Iterations of EM to run.')
    cl_args = parser.parse_args()

    run_extractors(doi_fname=cl_args.doi_file,
                   db_name=cl_args.db_name,
                   collection_name=cl_args.collection_name,
                   tar_task=cl_args.tar_task,
                   parsed_file_suffix=cl_args.parsed_file_suffix,
                   em_iters=cl_args.em_iters)


if __name__ == u'__main__':
    main()
