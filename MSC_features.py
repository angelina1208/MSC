import re

import nltk
from nltk.corpus import wordnet


class FeatureExtractor:

	def __init__(self) -> None:
		self._feature_extraction_methods = [regex_countable_features, nltk_countable_features()]
		self._list_of_feature_names = self.__list_of_feature_names()
		super().__init__()

	def __str__(self) -> str:
		return super().__str__()

	def __repr__(self) -> str:
		return super().__repr__()

	def text_to_feature_dictionary(self, text):
		all_features = {}
		for method in self._feature_extraction_methods:
			all_features.update(method(text))
		return all_features

	def list_of_feature_names(self):
		return self._list_of_feature_names

	def __list_of_feature_names(self):
		all_features = {}
		for method in self._feature_extraction_methods:
			all_features.update(method(""))
		return sorted(all_features.keys())




# https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
# Konvertiert POS-tags in wordnet tags
def _get_wordnet_pos(treebank_tag):
	if treebank_tag.startswith('J'):
		return wordnet.ADJ
	elif treebank_tag.startswith('V'):
		return wordnet.VERB
	elif treebank_tag.startswith('N'):
		return wordnet.NOUN
	elif treebank_tag.startswith('R'):
		return wordnet.ADV
	else:
		return ''


# Wrapt methode fuer tupel aus word und tag
def _wordnet_mapping(tuple):
	return tuple[0], _get_wordnet_pos(tuple[1])


def _average_length_of_words_in_iterable(words):
	num = 0.0
	for word in words:
		num += len(words)
	return num / len(words)


def regex_countable_features(text):
	letter = re.compile("")
	letter_upper = re.compile("")
	letter_lower = re.compile("")
	digits = re.compile("")
	whitespace = re.compile("")
	special = re.compile("")
	comma = re.compile("")
	dot = re.compile("")
	exclamation_mark = re.compile("")
	question_mark = re.compile("")
	colon = re.compile("")
	semicolon = re.compile("")
	hyphen = re.compile("")

	#TODO
	return {
		"num_char": len(text),
		"num_letter": len(re.findall(letter, text))
	}


def nltk_countable_features(text):
	sentences = nltk.sent_tokenize(text)
	words = nltk.word_tokenize(text)
	result = {
		"num_sentence": len(sentences),
		"num_words": len(words)
	}

	tagged_words = nltk.pos_tag(words)
	wordnet_tagged_words = map(_wordnet_mapping, tagged_words)

	result["num_adjectives"] = 0
	for tuple in wordnet_tagged_words:
		if tuple[1] == wordnet.ADJ:
			result["num_adjectives"] = result["num_adjectives"] + 1
		elif tuple[1] == wordnet.NOUN:
			pass #TODO
		else:
			pass #TODO

	return result
