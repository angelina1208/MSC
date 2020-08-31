# MSC_features.py
# Klasse, die Features aus den Movie Reviews extrahiert
# wird zum Training für den Movie Sentiment Classifier genutzt
#
# Angelina-Sophia Hauswald
# Matrikel-Nr.: 785803
# OS: Ubuntu 20.04 LTS 
# Python 3.8.2 (default, Apr 27 2020, 15:53:34)
# [GCC 9.3.0] on linux

import re
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.probability import FreqDist


class FeatureExtractor:
	def __init__(self, static_features=[]) -> None:
		# Liste aller Methodennamen, die genutzt werden, um Features zu extrahieren
		self._feature_extraction_methods = [_regex_countable_features, _nltk_countable_features, _vader_features]
		# Liste aller Features, die ausser den in den errechneten Features noch hinzugefuegt werden sollen, 
		# in diesem Fall nur der Goldstandard, d.h. ob eine Movie Review als pos oder neg gewertet wurde,
		# um abgleichen zu koennen
		self._expected_static_features = static_features
		# Liste aller Featurenamen (z.B "num_char", "num_letter")
		self._list_of_feature_names = self.__list_of_feature_names()
		super().__init__()


	def __str__(self) -> str:
		return "FeatureExtractor for features {} with static features {}.".format(self._list_of_feature_names,
																				  self._expected_static_features)

	def __repr__(self) -> str:
		return str(self)


	def features(self):
		return self._list_of_feature_names


	# gibt den Text weiter an alle Methoden, die die Features extrahieren
	# diese Methoden geben jeweils dictionaries zurueck mit ihren zugehörig errechneten Features
	# und werden in ein grosses dict zusammengefuehrt (all_features)
	# return: dict mit allen errechneten Features
	def text_to_feature_dictionary(self, text):
		all_features = {}
		for method in self._feature_extraction_methods:
			all_features.update(method(text))
		all_features.update(_ratio_features(all_features))
		return all_features


	# schreibt CSV, in der in jeder Zeile der Dateiname mit den zugehoerigen Werten der errechneten Features steht
	# list_file: 		Datei, in der alle Dateipfade angegeben sind, wo sich 
	# 					Trainings-, Validierungs- und Testdaten befinden
	# csv_file: 		Name der CSV, die erstellt bzw. beschrieben werden soll
	# append: 			wenn False, dann muss CSV erst erstellt werden und eine Header-Zeile angefertigt werden
	#					wenn True, dann existiert die CSV bereits, dementsprechen
	# 					existiert auch eine Header-Zeile und es muss nur noch hinzugefuegt werden
	# static_features:	dict, in dem angegeben ist, welches label der Goldstandard der aktuellen Datei hat
	def list_file_to_feature_csv(self, *, list_file, csv_file, append=False, static_features={}):
		n = 0
		# Kontrolle, ob für jedes statische Feature
		self.check_static_features_and_raise_error_if_needed(static_features)
		# alle zu bearbeitenden Dateien
		files = self.lines_of_file(list_file)
		# wenn CSV noch nicht vorhanden, neue CSV erstellen und Header-Zeile schreiben
		if not append:
			with open(csv_file, "w") as csv:
				csv.write("filename,")
				csv.write(",".join(self._list_of_feature_names) + "\n")
		with open(csv_file, "a") as csv:
			for file in files:
				with open(file, "r") as text_file:
					# gibt waehrend der Bearbeitung an bei der wievielten Datei sich das Programm befindet
					print("{}th file {}".format(n, file))
					n += 1
					# Datei lesen
					text = text_file.read()
				# aus dem Text alle Features ziehen
				feature_dict = self.text_to_feature_dictionary(text)
				# statische Features hinzufuegen
				feature_dict.update(static_features)
				# zusammenfassen der Features zu jeder Datei zu einer Zeile
				line = self.dict_and_filename_to_csv_line(feature_dict, file)
				# diese Zeile mit Features zu jeder Datei in CSV schreiben
				csv.write(line + "\n")


	# ueberprueft, ob die angegebenen statischen Features vorhanden sind und wirft sonst eine Fehlermeldung
	def check_static_features_and_raise_error_if_needed(self, static_features):
		for expected_static_feature in self._expected_static_features:
			if expected_static_feature not in static_features:
				raise Exception(
					"The static feature {} was defined at construction but is missing in static_features: {}".format(
						expected_static_feature, static_features))
		for static_feature in static_features:
			if static_feature not in self._expected_static_features:
				raise Exception(
					"The static feature {} is not listed in static features {}".format(static_feature, static_features))


	# fasst den Dateinamen und alle zu der Datei zugehoerigen berechneten Features in einen String zusammen
	# all_features: dict, in dem alle berechneten Features der Datei gespeichert sind
	# filename: Dateiname
	# return: Zeile bestehend aus Dateinamen und den Features
	def dict_and_filename_to_csv_line(self, all_features, filename):
		line = filename + ","
		for feature in self._list_of_feature_names:
			line = line + str(all_features[feature]) + ","
		line = line[0:-1]
		return line


	# file: aufgelistet, wo sich Trainings-, Validierungs- und Testdaten befinden
	# geht zeilenweise durch die Datei 
	# return: Dateipfad
	def lines_of_file(self, file):
		with open(file, "r") as list_file:
			files = list_file.readlines()
		# Entferne Zeilenumbrueche aus Eintraegen
		files = [x.strip() for x in files]
		return files


	# Funktion wird verwendet um erstmalig alle Namen der zukuenftig verwendeten Features
	# zu erstellen (und im Konstruktor in einer Klassenvariable zu speichern)
	# return: Liste aller Featurenamen
	def __list_of_feature_names(self):
		all_features = self.text_to_feature_dictionary(
			"Why don't you listen to https://www.youtube.com/watch?v=dQw4w9WgXcQ")
		#sortiert Features alphabetich
		feature_list = sorted(all_features.keys())
		feature_list += sorted(self._expected_static_features)
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


# return: durchschnittliche Wortlaenge
def _average_length_of_words_in_iterable(words):
	num = 0.0
	for word in words:
		num += len(word)
	return num / len(words)


# extrahiert character-based und punctuation-based Features
# return: dict mit den gezaehlten Features
def _regex_countable_features(text):
	letter = re.compile("[a-zöäüA-ZÖÄÜ]")  	# Buchstaben
	letter_upper = re.compile("[A-ZÖÄÜ]")  	# Grossbuchstaben
	letter_lower = re.compile("[a-zöäü]")  	# Kleinbuchstaben
	digits = re.compile("[0-9]")  			# Nummern
	whitespace = re.compile(" ")  			# Leerzeichen
	special = re.compile(r"[^\w]")  		# Sonderzeichen
	comma = re.compile(",")  				# Kommata
	dot = re.compile(r"\.")  				# Punkte
	exclamation_mark = re.compile("!")  	# Ausrufezeichen
	question_mark = re.compile(r"\?")  		# Fragezeichen
	colon = re.compile(":")  				# Doppelpunkte
	semicolon = re.compile(";")  			# Semicolon
	hyphen = re.compile("-")  				# Bindestrich

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
		"num_exclamationmark": len(re.findall(exclamation_mark, text)),
		"num_questionmark": len(re.findall(question_mark, text)),
		"num_colon": len(re.findall(colon, text)),
		"num_semicolon": len(re.findall(semicolon, text)),
		"num_hyphen": len(re.findall(hyphen, text))
	}

	return result


# extrahiert word-based Features und sentence-based Features
# return: dict mit den gezaehlten Features
def _nltk_countable_features(text):
	sentences = nltk.sent_tokenize(text)

	# entfernen aller special characters
	text_no_specials = re.sub(r'[^\w]', ' ', text)
	# in Kleinbuchstaben umwandeln
	text_no_specials_lower = text_no_specials.lower()
	# tokenisieren
	words = nltk.word_tokenize(text_no_specials_lower)
	# Menge aller Stoppwoerter im Englischen
	en_stops = set(stopwords.words('english'))
	result = {
		"num_sentence": len(sentences),
		"num_words": len(words),
		"num_stopwords": 0
	}
	# zaehlt die Anzahl der vorkommenden Stoppwoerter
	for word in words:
		if word in en_stops:
			result["num_stopwords"] += 1

	# gibt den Woertern die zugehoerigen POS
	tagged_words = nltk.pos_tag(words)
	wordnet_tagged_words = map(_wordnet_mapping, tagged_words)

	result["num_adjectives"] = 0
	result["num_nouns"] = 0
	result["num_verbs"] = 0
	result["num_adverbs"] = 0
	# zaehlt Anzahl an Adjektiven, Substantiven, Verben, Adverben
	for tuple in wordnet_tagged_words:
		if tuple[1] == wordnet.ADJ:
			result["num_adjectives"] = result["num_adjectives"] + 1
		elif tuple[1] == wordnet.NOUN:
			result["num_nouns"] = result["num_nouns"] + 1
		elif tuple[1] == wordnet.VERB:
			result["num_verbs"] = result["num_verbs"] + 1
		elif tuple[1] == wordnet.ADV:
			result["num_adverbs"] = result["num_adverbs"] + 1

	# zaehlt Anzahl an Token und Types
	result["num_tokens"] = len(words)
	types = nltk.Counter(words)
	result["num_types"] = len(types)
	# berechnet Hapax legomena
	hapaxes = FreqDist(nltk.Text(words)).hapaxes()
	result["num_hapaxes"] = len(hapaxes)

	return result

# errechnet den overall sentiment score vom Text
# return: dict mit dem berechneten sentiment score zum Text
def _vader_features(text):
	from nltk.sentiment.vader import SentimentIntensityAnalyzer
	analyzer = SentimentIntensityAnalyzer()
	vader = analyzer.polarity_scores(text)
	pos, neu, neg, comp = vader["pos"], vader["neu"], vader["neg"], vader["compound"]
	return {
		"vader_pos": pos,
		"vader_neu": neu,
		"vader_neg": neg,
		"vader_compound": comp
		# "vader_diff_pos_neg": pos-neg,
		# "vader_diff_pos_neu": pos-neu,
		# "vader_diff_neu_neg": neu-neg,
		# "vader_ratio_pos_neg": pos/neg,
		# "vader_ratio_pos_neu": pos/neu,
		# "vader_ratio_neu_neg": neu/neg
	}

# bisher wurden immer nur die Vorkommen von bestimmten Features ausgerechnet
# diese Funktion berechnet Verhältnisse von den Vorkommen, also bspw. die Type-Token-Ratio
# oder wieviele Punkte im Verhältnis zu allen Zeichen in einem Text vorkommen usw.
# return: dict mit den errechneten Features
def _ratio_features(feature_dict):
	ratio_dict = {}

	ratio_dict["ratio_type_token"] = 1 if feature_dict["num_tokens"] == 0 else \
		feature_dict["num_types"] / feature_dict["num_tokens"]
	ratio_dict["ratio_hapax_legomena_token"] = 1 if feature_dict["num_tokens"] == 0 else \
		feature_dict["num_hapaxes"] / feature_dict["num_tokens"]
	ratio_dict["ratio_char_dot"] = 1 if feature_dict["num_char"] == 0 else \
		feature_dict["num_dot"] / feature_dict["num_char"]
	ratio_dict["ratio_char_comma"] = 1 if feature_dict["num_char"] == 0 else \
		feature_dict["num_comma"] / feature_dict["num_char"]
	ratio_dict["ratio_char_semicolon"] = 1 if feature_dict["num_char"] == 0 else \
		feature_dict["num_semicolon"] / feature_dict["num_char"]
	ratio_dict["ratio_char_colon"] = 1 if feature_dict["num_char"] == 0 else \
		feature_dict["num_colon"] / feature_dict["num_char"]
	ratio_dict["ratio_char_exclamation"] = 1 if feature_dict["num_char"] == 0 else \
		feature_dict["num_exclamationmark"] / feature_dict["num_char"]
	ratio_dict["ratio_char_questionmark"] = 1 if feature_dict["num_char"] == 0 else \
		feature_dict["num_questionmark"] / feature_dict["num_char"]
	ratio_dict["ratio_char_hyphen"] = 1 if feature_dict["num_char"] == 0 else \
		feature_dict["num_hyphen"] / feature_dict["num_char"]
	ratio_dict["ratio_dot_comma"] = 1 if feature_dict["num_dot"] == 0 else \
		feature_dict["num_comma"] / feature_dict["num_dot"]
	ratio_dict["ratio_dot_questionmark"] = 1 if feature_dict["num_dot"] == 0 else \
		feature_dict["num_questionmark"] / feature_dict["num_dot"]
	ratio_dict["ratio_dot_exclamation"] = 1 if feature_dict["num_dot"] == 0 else \
		feature_dict["num_exclamationmark"] / feature_dict["num_dot"]
	ratio_dict["ratio_char_digits"] = 1 if feature_dict["num_char"] == 0 else \
		feature_dict["num_digits"] / feature_dict["num_char"]
	ratio_dict["ratio_char_special"] = 1 if feature_dict["num_char"] == 0 else \
		feature_dict["num_special"] / feature_dict["num_char"]
	ratio_dict["ratio_char_whitespace"] = 1 if feature_dict["num_char"] == 0 else \
		feature_dict["num_whitespace"] / feature_dict["num_char"]
	ratio_dict["ratio_letter_upper"] = 1 if feature_dict["num_char"] == 0 else \
		feature_dict["num_letter_upper"] / feature_dict["num_char"]
	ratio_dict["ratio_letter_lower"] = 1 if feature_dict["num_char"] == 0 else \
		feature_dict["num_letter_lower"] / feature_dict["num_char"]

	ratio_dict["ratio_sentences_dot"] = 1 if feature_dict["num_sentence"] == 0 else \
		feature_dict["num_dot"] / feature_dict["num_sentence"]
	ratio_dict["ratio_sentences_comma"] = 1 if feature_dict["num_sentence"] == 0 else \
		feature_dict["num_comma"] / feature_dict["num_sentence"]
	ratio_dict["ratio_sentences_digits"] = 1 if feature_dict["num_sentence"] == 0 else \
		feature_dict["num_digits"] / feature_dict["num_sentence"]
	ratio_dict["ratio_sentences_semicolon"] = 1 if feature_dict["num_sentence"] == 0 else \
		feature_dict["num_semicolon"] / feature_dict["num_sentence"]
	ratio_dict["ratio_sentences_questionmark"] = 1 if feature_dict["num_sentence"] == 0 else \
		feature_dict["num_questionmark"] / feature_dict["num_sentence"]
	ratio_dict["ratio_sentences_exclamation"] = 1 if feature_dict["num_sentence"] == 0 else \
		feature_dict["num_exclamationmark"] / feature_dict["num_sentence"]
	ratio_dict["ratio_sentences_hyphen"] = 1 if feature_dict["num_sentence"] == 0 else \
		feature_dict["num_hyphen"] / feature_dict["num_sentence"]
	ratio_dict["ratio_sentences_colon"] = 1 if feature_dict["num_sentence"] == 0 else \
		feature_dict["num_colon"] / feature_dict["num_sentence"]
	ratio_dict["ratio_sentences_char"] = 1 if feature_dict["num_sentence"] == 0 else \
		feature_dict["num_char"] / feature_dict["num_sentence"]
	ratio_dict["ratio_sentences_letter"] = 1 if feature_dict["num_sentence"] == 0 else \
		feature_dict["num_letter"] / feature_dict["num_sentence"]
	ratio_dict["ratio_sentences_letter_upper"] = 1 if feature_dict["num_sentence"] == 0 else \
		feature_dict["num_letter_upper"] / feature_dict["num_sentence"]
	ratio_dict["ratio_sentences_letter_lower"] = 1 if feature_dict["num_sentence"] == 0 else \
		feature_dict["num_letter_lower"] / feature_dict["num_sentence"]
	ratio_dict["ratio_sentences_special"] = 1 if feature_dict["num_sentence"] == 0 else \
		feature_dict["num_special"] / feature_dict["num_sentence"]
	ratio_dict["ratio_sentences_words"] = 1 if feature_dict["num_sentence"] == 0 else \
		feature_dict["num_words"] / feature_dict["num_sentence"]

	ratio_dict["ratio_words_stopwords"] = 1 if feature_dict["num_words"] == 0 else \
		feature_dict["num_stopwords"] / feature_dict["num_words"]
	ratio_dict["ratio_words_adjectives"] = 1 if feature_dict["num_words"] == 0 else \
		feature_dict["num_adjectives"] / feature_dict["num_words"]
	ratio_dict["ratio_words_verbs"] = 1 if feature_dict["num_words"] == 0 else \
		feature_dict["num_verbs"] / feature_dict["num_words"]
	ratio_dict["ratio_words_adverbs"] = 1 if feature_dict["num_words"] == 0 else \
		feature_dict["num_adverbs"] / feature_dict["num_words"]
	ratio_dict["ratio_words_nouns"] = 1 if feature_dict["num_words"] == 0 else \
		feature_dict["num_nouns"] / feature_dict["num_words"]
	ratio_dict["ratio_words_chars"] = 1 if feature_dict["num_words"] == 0 else \
		feature_dict["num_char"] / feature_dict["num_words"]
	return ratio_dict
