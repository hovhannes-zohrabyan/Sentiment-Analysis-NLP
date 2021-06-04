import nltk
from nltk.corpus import state_union
from nltk.tokenize import word_tokenize
from nltk.tag import DefaultTagger, UnigramTagger, BigramTagger, TrigramTagger
from nltk.corpus import treebank
from nltk.tag import RegexpTagger


class POSTagger:
    """
    POSTagger is class intended to create NLTK SequentialBackoffTagger for POS tagging.
    This Model gives ability to train custom tagger and if trained model fails backoffs to NLTK tagger
    """

    def __init__(self):
        self.custom_train_set = {}
        self.patterns = []

    # Method for creating a chain of POS taggers
    @staticmethod
    def backoff_tagger(tagger_classes, backoff=None, train_set=None):
        if train_set is None:
            # Choose amount of data samples fed to train
            train_set = treebank.tagged_sents()[:3000]

        for cls in tagger_classes:
            backoff = cls(train_set, backoff=backoff)

        return backoff

    def custom_tagger(self):
        custom_tagger = UnigramTagger(model=self.custom_train_set)

        return custom_tagger

    def regexp_tagger(self):
        tagger = RegexpTagger(self.patterns)

    def create_main_tagger(self):
        backoff = DefaultTagger('NN')
        custom_tagger = self.custom_tagger()

        chain_tagger = self.backoff_tagger([custom_tagger, UnigramTagger, BigramTagger, TrigramTagger], backoff=backoff)

        return chain_tagger


class POSProcessing:
    """
    POSProcessing is class intended to use tagger to select words that will proceed further.
    """
    def __init__(self):
        print("POS Processing model")

    @staticmethod
    def chunk_processing(sentence):
        tokenized = word_tokenize(sentence)
        try:
            for i in tokenized:
                words = nltk.word_tokenize(i)
                tagged = nltk.pos_tag(words)

                chunk_gram = r"""Chunk: {<.*>+?}
                                         }<VB.?|IN|DT>+{"""

                chunk_parser = nltk.RegexpParser(chunk_gram)
                chunked = chunk_parser.parse(tagged)

                chunked.draw()

        except Exception as e:
            print(str(e))


# POS tag list:
#
# CC	coordinating conjunction
# CD	cardinal digit
# DT	determiner
# EX	existential there (like: "there is" ... think of it like "there exists")
# FW	foreign word
# IN	preposition/subordinating conjunction
# JJ	adjective	'big'
# JJR	adjective, comparative	'bigger'
# JJS	adjective, superlative	'biggest'
# LS	list marker	1)
# MD	modal	could, will
# NN	noun, singular 'desk'
# NNS	noun plural	'desks'
# NNP	proper noun, singular	'Harrison'
# NNPS	proper noun, plural	'Americans'
# PDT	predeterminer	'all the kids'
# POS	possessive ending	parent\'s
# PRP	personal pronoun	I, he, she
# PRP$	possessive pronoun	my, his, hers
# RB	adverb	very, silently,
# RBR	adverb, comparative	better
# RBS	adverb, superlative	best
# RP	particle	give up
# TO	to	go 'to' the store.
# UH	interjection	errrrrrrrm
# VB	verb, base form	take
# VBD	verb, past tense	took
# VBG	verb, gerund/present participle	taking
# VBN	verb, past participle	taken
# VBP	verb, sing. present, non-3d	take
# VBZ	verb, 3rd person sing. present	takes
# WDT	wh-determiner	which
# WP	wh-pronoun	who, what
# WP$	possessive wh-pronoun	whose
# WRB	wh-abverb	where, when
