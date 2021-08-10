#!/usr/bin/env python
#
# File Name : ptbtokenizer.py
#
# Description : Do the PTB Tokenization and remove punctuations.
#
# Creation Date : 29-12-2014
# Last Modified : Thu Mar 19 09:53:35 2015
# Authors : Hao Fang <hfang@uw.edu> and Tsung-Yi Lin <tl483@cornell.edu>

import os
import subprocess
import tempfile
import nltk
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import pos_tag

class PTBTokenizer(object):
    """Python wrapper of Stanford PTBTokenizer"""

    corenlp_jar = 'stanford-corenlp-3.4.1.jar'
    punctuations = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", \
                    ".", "?", "!", ",", ":", "-", "--", "...", ";"]

    @classmethod
    def tokenize(cls, corpus):
        cmd = ['java', '-cp', cls.corenlp_jar, \
                'edu.stanford.nlp.process.PTBTokenizer', \
                '-preserveLines', '-lowerCase']

        if isinstance(corpus, list) or isinstance(corpus, tuple):
            if isinstance(corpus[0], list) or isinstance(corpus[0], tuple):
                corpus = {i:c for i, c in enumerate(corpus)}
            else:
                corpus = {i: [c, ] for i, c in enumerate(corpus)}

        # prepare data for PTB Tokenizer
        tokenized_corpus = {}
        image_id = [k for k, v in list(corpus.items()) for _ in range(len(v))]
        sentences = '\n'.join([c.replace('\n', ' ') for k, v in corpus.items() for c in v])

        #TODO: tokenizer
        text = "".join([char for char in sentences if char not in cls.punctuations])
        sent_text = nltk.sent_tokenize(text)
        stop_words = stopwords.words('english')
        porter = PorterStemmer()
        new_sentences = []

        for sentence in sent_text:
            words = word_tokenize(sentence)
            filtered_words = [word for word in words if word not in stop_words]
            stemmed = [porter.stem(word) for word in filtered_words]
            new_sentence = ' '.join(stemmed)
            new_sentences.append(new_sentence)

        #sentences = '\n'.join(new_sentences)
        lines = new_sentences#sentences.split('\n')
        print(lines)
        # create dictionary for tokenized captions
        for k, line in zip(image_id, lines):
            if not k in tokenized_corpus:
                tokenized_corpus[k] = []
            tokenized_caption = ' '.join([w for w in line.rstrip().split(' ') \
                                          if w not in cls.punctuations])
            tokenized_corpus[k].append(tokenized_caption)

        return tokenized_corpus