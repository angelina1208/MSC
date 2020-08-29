import re

import nltk
from nltk.corpus import wordnet, stopwords
from nltk.probability import FreqDist

class FeatureExtractor:

	def __init__(self, static_features) -> None:
		self._feature_extraction_methods = [_regex_countable_features, _nltk_countable_features]
		self._static_features = static_features
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

	def list_file_to_feature_csv(self, *, list_file, csv_file, append=False, static_features={}):
		n = 0
		for extepcted_static_feature in self._static_features:
			if extepcted_static_feature not in static_features:
				raise Exception(
					"The static feature {} was defined at construction but is missing in static_features: {}".format(
						extepcted_static_feature, static_features))
		files = self.lines_of_file(list_file)
		if not append:
			with open(csv_file, "w") as csv:
				csv.write(",".join(self._list_of_feature_names) + "\n")
		with open(csv_file, "a") as csv:
			for file in files:
				with open(file, "r") as text_file:
					print("{}th file {}".format(n, file))
					n += 1
					text = text_file.read()
				feature_dict = self.text_to_feature_dictionary(text)
				feature_dict.update(static_features)
				line = self.dict_and_filename_to_csv_line(feature_dict, file)
				csv.write(line + "\n")

	def dict_and_filename_to_csv_line(self, all_features, filename):
		line = filename + ","
		for feature in self._list_of_feature_names:
			line = line + str(all_features[feature]) + ","
		line = line[0:-1]
		return line

	def lines_of_file(self, file):
		with open(file, "r") as list_file:
			files = list_file.readlines()
		# Entferne zeilenumbrueche aus eintraegen
		files = [x.strip() for x in files]
		return files

	def __list_of_feature_names(self):
		all_features = {}
		for method in self._feature_extraction_methods:
			all_features.update(method("Why don't you listen to https://www.youtube.com/watch?v=dQw4w9WgXcQ"))
		feature_list = sorted(all_features.keys())
		feature_list += sorted(self._static_features)
		return feature_list


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
		num += len(word)
	return num / len(words)


def _regex_countable_features(text):
	letter = re.compile("[a-zöäüA-ZÖÄÜ]")
	letter_upper = re.compile("[A-ZÖÄÜ]")
	letter_lower = re.compile("[a-zöäü]")
	digits = re.compile("[0-9]")
	whitespace = re.compile(" ")
	special = re.compile("""[^\w]""")
	comma = re.compile(",")
	dot = re.compile("""\.""")
	exclamation_mark = re.compile("!")
	question_mark = re.compile("""\?""")
	colon = re.compile(":")
	semicolon = re.compile(";")
	hyphen = re.compile("-")

	result = {
		"num_char": len(text),
		"num_letter": len(re.findall(letter, text)),
		"num_letter_upper": len(re.findall(letter_upper, text)),
		"num_letter_lower": len(re.findall(letter_lower, text)),
		"num_digits": len(re.findall(digits, text)),
		"num_whitespace": len(re.findall(whitespace, text)),
		"num_special": len(re.findall(special, text)),
		"num_comma": len(re.findall(comma, text)),
		"num_dot": len(re.findall(dot, text)),
		"num_exclamation_mark": len(re.findall(exclamation_mark, text)),
		"num_question_mark": len(re.findall(question_mark, text)),
		"num_colon": len(re.findall(colon, text)),
		"num_semicolon": len(re.findall(semicolon, text)),
		"num_hyphen": len(re.findall(hyphen, text))
	}

	return result


def _nltk_countable_features(text):
	sentences = nltk.sent_tokenize(text)

	# entfernen aller special characters
	text_no_specials = re.sub(r'[^\w]', ' ', text)
	text_no_specials_lower = text_no_specials.lower()
	words = nltk.word_tokenize(text_no_specials_lower)
	en_stops = set(stopwords.words('english'))
	result = {
		"num_sentence": len(sentences),
		"num_words": len(words),
		"num_stopwords": 0
	}
	for word in words:
		if word in en_stops:
			result["num_stopwords"] += 1

	tagged_words = nltk.pos_tag(words)
	wordnet_tagged_words = map(_wordnet_mapping, tagged_words)

	result["num_adjectives"] = 0
	result["num_nouns"] = 0
	result["num_verb"] = 0
	result["num_adverbs"] = 0
	for tuple in wordnet_tagged_words:
		if tuple[1] == wordnet.ADJ:
			result["num_adjectives"] = result["num_adjectives"] + 1
		elif tuple[1] == wordnet.NOUN:
			result["num_nouns"] = result["num_nouns"] + 1
		elif tuple[1] == wordnet.VERB:
			result["num_verb"] = result["num_verb"] + 1
		elif tuple[1] == wordnet.ADV:
			result["num_adverbs"] = result["num_adverbs"] + 1

	tokens = nltk.word_tokenize(text_no_specials_lower)
	types = nltk.Counter(tokens)

	ttr = len(types) / len(tokens)
	result["num_types"] = len(types)
	result["ratio_type_token"] = ttr

	hapaxes = FreqDist(nltk.Text(tokens)).hapaxes()
	result["num_hapaxes"] = len(hapaxes)
	result["ratio_hapax_legomena_token"] = len(hapaxes) / len(tokens)

	return result
