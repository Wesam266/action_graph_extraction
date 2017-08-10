from __future__ import unicode_literals
import os, sys, time
import codecs, argparse

import agex_settings
import graph_extractor


def run_extractor(db_name, collection_name, tar_task, parsed_file_suffix,
                  doi_file, model_dir, em_iters, term_min_swaps):
    AGE = graph_extractor.ActionGraphExtractor()
    # Load training data and initialize connections sequentially.
    with codecs.open(doi_file, 'r') as doi_file_fh:
        AGE.load_parsed_recipes(doi_file=doi_file_fh, db_name=db_name,
                                collection_name=collection_name,
                                tar_task=tar_task,
                                parsed_file_suffix=parsed_file_suffix)

    # Do em_iters iterations of E and M.
    em_start_time = time.time()
    performed_iters = 0
    for i in range(em_iters):
        print(u'EM iteration:{}'.format(i))
        iter_start = time.time()
        try:
            AGE.M_step_all()
            num_changes = AGE.local_search_all()
        # https://stackoverflow.com/q/16123529/3262406
        except KeyboardInterrupt as ke:
            raise
        except Exception as e:
            print('ERROR: ' + unicode(e))
            # If things go bad atleast save your stuff :-P
            # Save all models to disk.
            AGE.save_all_models(model_dir=model_dir, extra_suffix='-error')
            # Save all learnt graphs to disk. The directory needs to
            # exist already.
            learnt_graph_dest = os.path.join(agex_settings.in_data_dir,
                                             u'{:s}-{:s}-{:s}-results'.format(
                                                 db_name,
                                                 collection_name,
                                                 tar_task))
            AGE.save_learnt_nxgraphs(dest_dir=learnt_graph_dest,
                                     graph_suffix='uwkiddon-error')
            print('Exiting in error.')
            sys.exit(1)
        iter_end = time.time()
        print(u'Local search edge swaps.: {:d}'.format(num_changes))
        print('Iteration {:d} of EM in {:.3f}s'.format(i, iter_end-iter_start))
        performed_iters += 1
        if num_changes < term_min_swaps:
            print('Fewer than {:d} edge swaps.; exiting.'.format(term_min_swaps))
            break
    em_end_time = time.time()
    print('{:d} iterations of EM in {:.3f}s'.format(performed_iters,
                                                    em_end_time-em_start_time))

    # Save all models to disk.
    AGE.save_all_models(model_dir=model_dir)
    # Save all learnt graphs to disk.
    learnt_graph_dest = os.path.join(agex_settings.in_data_dir,
                                     u'{:s}-{:s}-{:s}-results'.format(db_name,
                                                                      collection_name,
                                                                      tar_task))
    AGE.save_learnt_nxgraphs(dest_dir=learnt_graph_dest,
                             graph_suffix='uwkiddon')

    # Print models and graphs learnt if you want, to debug :-P
    if False:
        AGE.print_all_models()
        for i, AG in enumerate(AGE.actionGraphs):
            print(u'\n\n\nAction graph {}'.format(i))
            # Not using this unicode thing here gives me some very annoying errors.
            print(unicode(AG))


def run_baseline(db_name, collection_name, tar_task, parsed_file_suffix,
                 doi_file, model_dir):
    """
    Run the baseline model which does a sequential initialization and saves
    graphs as nx graphs to disk.
    :param doi_file:
    :param db_name:
    :param collection_name:
    :param tar_task:
    :param parsed_file_suffix:
    :return:
    """
    AGE = graph_extractor.ActionGraphExtractor()
    # Load training data and initialize connections sequentially.
    with codecs.open(doi_file, 'r') as doi_file_fh:
        AGE.load_parsed_recipes(doi_file=doi_file_fh, db_name=db_name,
                                collection_name=collection_name,
                                tar_task=tar_task,
                                parsed_file_suffix=parsed_file_suffix)

    # Update all parameters based on present state of graphs.
    AGE.M_step_all()

    # Save all models to disk.
    AGE.save_all_models(model_dir=model_dir)
    # Save all learnt graphs to disk.
    learnt_graph_dest = os.path.join(agex_settings.in_data_dir,
                                     u'{:s}-{:s}-{:s}-results'.format(db_name,
                                                              collection_name,
                                                              tar_task))
    AGE.save_learnt_nxgraphs(dest_dir=learnt_graph_dest,
                             graph_suffix='seq')


def main():
    """
    Parse all command line args and call appropriate functions.
    :return:
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest=u'subcommand',
                                       help=u'Model to run.')

    # Sequentially initialized model. The baseline.
    seq_model_parser = subparsers.add_parser(u'seq_model')
    seq_model_parser.add_argument(u'-d', u'--db_name',
                                  choices=[u'predsynth'],
                                  default=u'predsynth',
                                  help=u'Name of database data came from.')
    seq_model_parser.add_argument(u'-c', u'--collection_name',
                                  choices=[u'annotated_papers', u'papers'],
                                  default=u'annotated_papers',
                                  help=u'Mongo collection data came from.')
    seq_model_parser.add_argument(u'-t', u'--tar_task',
                                  choices=['age'],
                                  default='age',
                                  help=u'Target task of data; says which '
                                       u'directory to read')
    seq_model_parser.add_argument(u'-s', u'--parsed_file_suffix',
                                  required=True,
                                  choices=['deps_heu_parsed'],
                                  help=u'Suffix for the parsed json to read.')
    seq_model_parser.add_argument(u'-f', u'--doi_file',
                                  required=True,
                                  help=u'Path to text file with DOIs.')
    seq_model_parser.add_argument('-md', '--model_dir',
                                  required=True,
                                  help='The directory to which the learnt'
                                       ' models will get written.')

    # The full probabilistic model as in Kiddons papers.
    prob_model_parser = subparsers.add_parser(u'uwkiddon_model')
    prob_model_parser.add_argument(u'-d', u'--db_name',
                                   choices=[u'predsynth'],
                                   default=u'predsynth',
                                   help=u'Name of database data came from.')
    prob_model_parser.add_argument(u'-c', u'--collection_name',
                                   choices=[u'annotated_papers', u'papers'],
                                   default=u'annotated_papers',
                                   help=u'Mongo collection data came from.')
    prob_model_parser.add_argument(u'-t', u'--tar_task',
                                   choices=['age'],
                                   default='age',
                                   help=u'Target task of data; says which '
                                        u'directory to read')
    prob_model_parser.add_argument(u'-s', u'--parsed_file_suffix',
                                   required=True,
                                   choices=['deps_heu_parsed'],
                                   help=u'Suffix for the parsed json to read.')
    prob_model_parser.add_argument(u'-f', u'--doi_file',
                                   required=True,
                                   help=u'Path to text file with DOIs.')
    prob_model_parser.add_argument('-md', '--model_dir', required=True,
                                   help='The directory to which the learnt'
                                        ' models will get written.')
    prob_model_parser.add_argument(u'--em_iters', type=int, required=True,
                                   help=u'Iterations of EM to run.')
    # Im setting this kind of as 0.01*(200 graphs)*(10 ops)*(2 args)=40
    # But this needs to be more principled. :/
    prob_model_parser.add_argument(u'--term_min_swaps', type=int, required=True,
                                   help=u'Min number of swaps an EM iteration '
                                        u'must make. If lesser than this '
                                        u'learning stops.')

    cl_args = parser.parse_args()

    if cl_args.subcommand == u'seq_model':
        run_baseline(db_name=cl_args.db_name,
                     collection_name=cl_args.collection_name,
                     tar_task=cl_args.tar_task,
                     parsed_file_suffix=cl_args.parsed_file_suffix,
                     doi_file=cl_args.doi_file,
                     model_dir=cl_args.model_dir)
    elif cl_args.subcommand == u'uwkiddon_model':
        run_extractor(db_name=cl_args.db_name,
                      collection_name=cl_args.collection_name,
                      tar_task=cl_args.tar_task,
                      parsed_file_suffix=cl_args.parsed_file_suffix,
                      doi_file=cl_args.doi_file,
                      model_dir=cl_args.model_dir,
                      em_iters=cl_args.em_iters,
                      term_min_swaps=cl_args.term_min_swaps)

if __name__ == u'__main__':
    main()
