import os
import nltk
import constants

import graph_extractor



def main():
    AGE = graph_extractor.ActionGraphExtractor()
    AGE.load_train_file(constants.TRAIN_FILE)
    for i in range(10):
        AGE.M_step_all()
        print "======================================================================================================"
        num_changes = AGE.local_search_all()
        if num_changes < 2: #
            break

    for AG in AGE.actionGraphs:
        print AG


if  __name__ == '__main__':
    main()