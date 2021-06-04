import re
from nltk.corpus import wordnet
import enchant
from nltk.metrics import edit_distance

# !!! Important --> The above two functions have to be used before tokenizing


class RegExpReplacer:
    """
    RegExpReplacer is class intended to find occurrences with regexp patterns and replace them using given dictionary
    This tool can be used to replace contractions with their expanded forms (can't -> cannot)
    """

    def __init__(self):
        self.replacement_patterns = [(r'won\'t', 'will not'), (r'can\'t', 'cannot'),
                                     (r'i\'m', 'i am'), (r'ain\'t', 'is not'),
                                     (r'(\w+)\'ll', '\g<1> will'), (r'(\w+)n\'t', '\g<1> not'),
                                     (r'(\w+)\'ve', '\g<1> have'), (r'(\w+)\'s', '\g<1> is'),
                                     (r'(\w+)\'re', '\g<1> are'), (r'(\w+)\'d', '\g<1> would')
                                     ]
        self.patterns = [(re.compile(regexp), replacement) for (regexp, replacement) in self.replacement_patterns]
        
    def replace(self, text):
        to_replace = text
        for (pattern, replacement) in self.patterns:
            to_replace = re.sub(pattern, replacement, to_replace)
        return to_replace # noqa


class RemoveRepeatingCharacters:
    """
    RemoveRepeatingCharacters is class intended to find words with repeating letters and remove them to get the stem
    This tool can be used to replace long words with their short form (loooove -> love | ooooooh -> oh)
    """

    def __init__(self):
        self.repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
        self.repl = r'\1\2\3'

    def replace(self, word, wordnet_check_allowed=True):
        # WordNet check is very time consuming function, however having word checked before making any changes
        # Is a lot more accurate
        if wordnet_check_allowed:
            if wordnet.synsets(word):
                return word

        replacement_word = self.repeat_regexp.sub(self.repl, word)
        if replacement_word != word:
            return self.replace(replacement_word)
        else:
            return replacement_word


class SpellingCorrection:
    """
    RegExpReplacer is class intended to find misspellings in text and replace them with
    right word using Aspell(http://aspell.net/.) and Enchant
    This tool can be used for spell checking (cookbok -> cookbook)
    """

    def __init__(self, dict_name="en", max_dist=2):
        self.spell_dict = enchant.Dict(dict_name)
        self.max_dist = max_dist

    def correct(self, word):
        if self.spell_dict.check(word):
            return word

        suggestion = self.spell_dict.suggest(word)

        # Check for suggestion not to be too far from the actual word
        if suggestion and edit_distance(word, suggestion[0]) <= self.max_dist:
            return suggestion[0]
        else:
            return word
