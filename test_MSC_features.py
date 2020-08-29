from MSC_features import _ratio_features, _vader_features, _regex_countable_features, _nltk_countable_features


def test_vader_features_sample():
	features = _vader_features("Test")
	assert len(features) == 4


def test_vader_features_empty():
	features = _vader_features("")
	assert len(features) == 4


def test_regex_countable_features_empty():
	features = _regex_countable_features("")
	assert len(features) > 0
	for feature in features:
		assert feature.startswith("num")
	assert features["num_char"] == 0
	assert features["num_letter"] == 0
	assert features["num_letter_upper"] == 0
	assert features["num_letter_lower"] == 0
	assert features["num_digits"] == 0
	assert features["num_whitespace"] == 0
	assert features["num_special"] == 0
	assert features["num_comma"] == 0
	assert features["num_dot"] == 0
	assert features["num_exclamationmark"] == 0
	assert features["num_questionmark"] == 0
	assert features["num_colon"] == 0
	assert features["num_semicolon"] == 0
	assert features["num_hyphen"] == 0


def test_regex_countable_features_sample():
	features = _regex_countable_features("Hi. I äm a example,.- t€xt123.;:?!")
	assert len(features) > 0
	assert features["num_char"] == 34
	assert features["num_letter"] == 16
	assert features["num_letter_upper"] == 2
	assert features["num_letter_lower"] == 14
	assert features["num_digits"] == 3
	assert features["num_whitespace"] == 5
	assert features["num_special"] == 15
	assert features["num_comma"] == 1
	assert features["num_dot"] == 3
	assert features["num_exclamationmark"] == 1
	assert features["num_questionmark"] == 1
	assert features["num_colon"] == 1
	assert features["num_semicolon"] == 1
	assert features["num_hyphen"] == 1


def test_nltk_countable_features_empty():
	features = _nltk_countable_features("")
	assert len(features) > 0
	for feature in features:
		assert feature.startswith("num")

	assert features["num_sentence"] == 0
	assert features["num_words"] == 0
	assert features["num_stopwords"] == 0
	assert features["num_adjectives"] == 0
	assert features["num_nouns"] == 0
	assert features["num_verbs"] == 0
	assert features["num_adverbs"] == 0
	assert features["num_tokens"] == 0
	assert features["num_types"] == 0
	assert features["num_hapaxes"] == 0


def test_nltk_countable_features_sample():
	features = _nltk_countable_features("I am a example sentence. There should be many interesting known things.")
	assert features["num_sentence"] == 2
	assert features["num_words"] == 12
	assert features["num_stopwords"] == 6
	assert features["num_adjectives"] == 2
	assert features["num_nouns"] == 4
	assert features["num_verbs"] == 3
	assert features["num_adverbs"] == 0
	assert features["num_tokens"] == 12
	assert features["num_types"] == 12
	assert features["num_hapaxes"] == 12

def test_ratio_features_empty():
	features = {}
	features["num_char"] = 0
	features["num_letter"] = 0
	features["num_letter_upper"] = 0
	features["num_letter_lower"] = 0
	features["num_digits"] = 0
	features["num_whitespace"] = 0
	features["num_special"] = 0
	features["num_comma"] = 0
	features["num_dot"] = 0
	features["num_exclamationmark"] = 0
	features["num_questionmark"] = 0
	features["num_colon"] = 0
	features["num_semicolon"] = 0
	features["num_hyphen"] = 0
	features["num_sentence"] = 0
	features["num_words"] = 0
	features["num_stopwords"] = 0
	features["num_adjectives"] = 0
	features["num_nouns"] = 0
	features["num_verbs"] = 0
	features["num_adverbs"] = 0
	features["num_tokens"] = 0
	features["num_types"] = 0
	features["num_hapaxes"] = 0

	new_features = _ratio_features(features)
	for value in new_features.values():
		assert value == 1
	for feature in new_features:
		assert feature.startswith("ratio")