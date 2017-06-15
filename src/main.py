import os
import nltk
import constants

import graph_extractor



def main():
    AGE = graph_extractor.ActionGraphExtractor()
    # Load train file and initialize connections sequentially.
    AGE.load_train_file(constants.TRAIN_FILE_FEAT)
    for i in range(1):
        print('\n\n\nIteration:{}'.format(i))
        AGE.M_step_all()
        # num_changes = AGE.local_search_all()
        # print('num_changes: {:d}'.format(num_changes))
        # if num_changes < 2:
        #     break

    # for i, AG in enumerate(AGE.actionGraphs):
    #     print('\n\n\nAction graph {}'.format(i))
    #     print AG


if  __name__ == '__main__':
    main()