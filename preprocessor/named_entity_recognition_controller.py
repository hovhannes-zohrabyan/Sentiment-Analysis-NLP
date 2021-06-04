import nltk
from nltk.corpus import state_union
from nltk.tokenize import sent_tokenize
from nltk.tokenize import PunktSentenceTokenizer

sample_text = state_union.raw("2006-GWBush.txt")
train_text = state_union.raw("2005-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)


class NamedEntityRecognitionController:

    @staticmethod
    def process_words(words):
        word_set = nltk.word_tokenize(words)
        tagged = nltk.pos_tag(word_set)

        named_ent = nltk.ne_chunk(tagged, binary=True)

        return named_ent

    # TODO: Add custom sent_tokenizer
    def process_strings(self, sentence):
        words = sent_tokenize(sentence)
        named_ent = self.process_words(words)
        return named_ent

    def show_entities(self, words):
        named_ent = self.process_words(words)
        named_ent.draw()

# Named Entity Type and Examples
#
# ORGANIZATION - Georgia-Pacific Corp., WHO
# PERSON - Eddy Bonte, President Obama
# LOCATION - Murray River, Mount Everest
# DATE - June, 2008-06-29
# TIME - two fifty a m, 1:30 p.m.
# MONEY - 175 million Canadian Dollars, GBP 10.40
# PERCENT - twenty pct, 18.75 %
# FACILITY - Washington Monument, Stonehenge
# GPE - South East Asia, Midlothian
