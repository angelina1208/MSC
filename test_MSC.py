import pytest

LONG_TEXT = """Some of these features sound really basic, but don’t let their basic nature fool you! This
piece on punctuation in novels shows you how informative punctuation can be. And
here is an excellent quote by Gary Provost (from 100 Ways To Improve Your Writing)
that shows the power of sentence length to change the character of a text:
VARY SENTENCE LENGTH
This sentence has five words. Here are five more words. Five-word sentences
are fine. But several together become monotonous. Listen to what is hap-
pening. The writing is getting boring. The sound of it drones. It’s like a
stuck record. The ear demands some variety.
Now listen. I vary the sentence length, and I create music. Music. The
writing sings. It has a pleasant rhythm, a lilt, a harmony. I use short
sentences. And I use sentences of medium length. And sometimes when I am
certain the reader is rested, I will engage him with a sentence of considerable
length, a sentence that burns with energy and builds with all the impetus of
a crescendo, the roll of the drums, the crash of the cymbals–sounds that say
listen to this, it is important.
So write with a combination of short, medium, and long sentences. Create a
sound that pleases the reader’s ear. Don’t just write words. Write music.
Step 5: Store the computed statistics in a useful format (CSV/TSV)
As an example for this step, let’s say we are dealing with the author identification task
and we want to distinguish between the authors Jane Austen, Charlotte Bronte and
George Eliot. Let’s consider that we have two books from each author that we compute
the statistics on, and that we compute two statistics: average number of tokens per
paragraph and average sentence length. After computing the statistics, you can store
them on two levels:
91. Statistics for single instances
For each individual text, store your computed features and the gold class (e.g. the
author for authorship attribution). We would then store six rows, each representing
a book. For each row, we would have two numbers, one for each of the statistics,
and the gold class (which of the three authors this book belongs to). This is
especially useful for classifying your test instances in the end (see steps 7–8).
2. Aggregated statistics for each prediction class
For each class, store the aggregated statistics/features. In our example, we would
store three rows, each representing an author. For each row, we would store two
numbers, one for each of the statistics. This could especially be useful for creating
your model based on the training set (see step 7).
Step 6: Present the statistics to the user by means of visualization
(optional)
For this purpose, create some visualizations of the statistics you inferred on the training
set. A simple example of a visualization would be a graph that shows the number of
tokens per chapter for each of the books. This step is optional but can help you and the
reader to understand the data and/or approach. You can include useful visualizations
in your report.
Step 7: Use the computed statistics as features for an automatic classi-
fication task
By computing a set of statistics, you are already halfway towards building a classifier.
Let’s look at the other half. It is common in NLP to combine computed statistics in a
machine learning decision making system. Since we are not teaching machine learning
in this course, we suggest a different method for automatic classification.
Namely, given the values for N features, we want to compute the best match. This is
achieved in two steps:
1. Create a model of the classes you have encountered in the training set (you created
this already in the steps 4 and 5 of this assignment). This model tells us what is
the representative number of tokens, paragraph length, type-token ratio, etc. for
each class.
2. For each test case, you compute the values for the same N features, and you
compare their value to each of the known classes to determine the best match.
So, in our example, a test case would have the features f t,1 and f t,2 . We can
check how different this test case is from each known author, by computing its
feature distance from the features of each of the three au"""

from MSC_features import FeatureExtractor, _ratio_features, _vader_features, _regex_countable_features, \
	_nltk_countable_features


def test_vader_features_empty():
	features = _vader_features("")
	assert len(features) == 4


def test_vader_features_sample():
	features = _vader_features("Test")
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


def test_FeatureExtractor_init():
	FeatureExtractor()
	FeatureExtractor(["a static feature"])


def test_FeatureExtractor_missing_static_feature():
	missing_feature = "a static feature"
	fe = FeatureExtractor([missing_feature])
	with pytest.raises(Exception) as e_info:
		fe.list_file_to_feature_csv(list_file="idc", csv_file="idc")
		assert "static" in e_info
		assert "feature" in e_info
		assert missing_feature in e_info


def test_FeatureExtractor_unexpected_static_feature():
	unexpected_feature = "a static feature"
	fe = FeatureExtractor()
	with pytest.raises(Exception) as e_info:
		fe.list_file_to_feature_csv(list_file="idc", csv_file="idc", static_features={unexpected_feature: 0})
		assert "static" in e_info
		assert "feature" in e_info
		assert unexpected_feature in e_info


def test_FeatureExtractor_text_to_feature_dictionary_empty():
	fe = FeatureExtractor()
	features = fe.text_to_feature_dictionary("")
	for feature in features:
		if feature.startswith("num"):
			assert features[feature] == 0
		if feature.startswith("ratio"):
			assert features[feature] == 1
	assert features


def test_FeatureExtractor_text_to_feature_dictionary_long_text():
	fe = FeatureExtractor()
	features = fe.text_to_feature_dictionary(LONG_TEXT)
	assert features
