import os, sys
import nltk
import constants

import graph_extractor



def main():
    AGE = graph_extractor.ActionGraphExtractor()
    # Load train file and initialize connections sequentially.
    AGE.load_train_file(constants.TRAIN_FILE_FEAT)

    # Do 5 iterations of E and M.
    for i in range(5):
        print('\n\n\nEM iteration:{}'.format(i))
        AGE.M_step_all()
        num_changes = AGE.local_search_all()
        print('num_changes: {:d}'.format(num_changes))
        if num_changes < 2:
            break

    # Print models and graphs learnt.
    AGE.print_all_models()
    for i, AG in enumerate(AGE.actionGraphs):
        print('\n\n\nAction graph {}'.format(i))
        print AG


if  __name__ == '__main__':
    main()