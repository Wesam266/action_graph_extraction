import os
import nltk
import constants

import graph_extractor



def main():
    AGE = graph_extractor.ActionGraphExtractor()
    AGE.load_train_file(constants.TRAIN_FILE)
    AGE.M_step_all()
    print AGE.local_search_all()
    # AGE.M_step_all()
    # print AGE.local_search_all()


    for AG in AGE.actionGraphs:
        print AG


if  __name__ == '__main__':
    main()