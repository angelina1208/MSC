import nltk

# Need those downloads!
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
import re
# https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
# The following function would map the treebank tags to WordNet part of speech names:
from nltk.corpus import wordnet


def get_wordnet_pos(treebank_tag):
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


def wordnet_mapping(tuple):
	return tuple[0], get_wordnet_pos(tuple[1])


def average_length_of_words_in_iterable(words):
	num = 0.0
	for word in words:
		num += len(words)
	return num / len(words)


def counting_stuff(text):
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

	return {
		"num_char": len(text),
		"num_letter": len(re.findall(letter, text))
	}


def other_stuff(text):
	sentences = nltk.sent_tokenize(text)
	words = nltk.word_tokenize(text)
	result = {
		"num_sentence": len(sentences),
		"num_words": len(words)
	}

	tagged_words = nltk.pos_tag(words)
	wordnet_tagged_words = map(wordnet_mapping, tagged_words)

	result["num_adjectives"] = 0
	for tuple in wordnet_tagged_words:
		if tuple[1] == wordnet.ADJ:
			result["num_adjectives"] = result["num_adjectives"] + 1
		elif tuple[1] == wordnet.NOUN:
			pass
		else:
			pass

	return result


if __name__ == '__main__':
	print("hi")
	with open("aclImdb_v1/aclImdb/train/pos/0_9.txt") as file:
		text_pos = file.read()
	print(text_pos)
	with open("aclImdb_v1/aclImdb/train/neg/0_3.txt") as file:
		text_neg = file.read()
	print(text_neg)
