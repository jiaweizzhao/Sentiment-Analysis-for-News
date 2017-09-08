#!/usr/bin/env tensorflow

import word2vec

word2vec.word2vec('corpusSegDone.txt', 'corpusWord2Vec.bin', size=100,verbose=True)