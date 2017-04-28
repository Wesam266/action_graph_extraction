import os
import nltk

TRAIN_FILE='data/exp.txt'
'''
 1)word, 2)target label, 3)index, 4)lemma, 5)stemming, 6)POS tag, 7)NER tag, 8)9)10)3 columns of dependency tree info,
     11)boolean of stop word, 12)-19)8 columns of lexicon category counts, 20)boolean of ending with -ing,
     21)boolean of containing digit, 22)boolean of all digit, 23)boolean of containing uppercase, 24) boolean of all uppercase
'''
annot = []
for line in open(TRAIN_FILE):
    split = line.strip().split("\t")
    annot.append(split)
    print split




