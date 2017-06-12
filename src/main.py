import os
import nltk
import constants

import graph_extractor



def main():
    AGE = graph_extractor.ActionGraphExtractor()
    AGE.load_train_file(constants.TRAIN_FILE)
    AGE.M_step_all()
    print "====================================="
    print AGE.local_search_all()

    for i, AG in enumerate(AGE.actionGraphs):
        print('\n\n\nAction graph {}'.format(i))
        print AG


if  __name__ == '__main__':
    main()