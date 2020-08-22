import os

import pandas
import nltk

from MSC_features import regex_countable_features, nltk_countable_features


def ask_for_downloads():
	print("Do you need to download stuff?")
	nltk.download('punkt')
	nltk.download('averaged_perceptron_tagger')
	nltk.download('wordnet')
	print("You downloaded stuff.")


if __name__ == '__main__':
	print("Welcome to MSC.")
	ask_for_downloads()
	pos_folder = "aclImdb_v1/aclImdb/train/pos"
	neg_folder = "aclImdb_v1/aclImdb/train/neg"
	feature_store_file = "feats.csv"
	print("Training with positive samples from: {}\nand negative samples from: {}".format(pos_folder, neg_folder))
	if os.path.isfile(feature_store_file):
		os.remove(feature_store_file)
	has_headers = False
	n = 0
	for filename in os.listdir(pos_folder):
		with open(os.path.join(os.getcwd(), pos_folder, filename), 'r') as f:
			text = f.read()

		all_features = {}
		all_features.update(regex_countable_features(text))
		all_features.update(nltk_countable_features(text))
		keys_sorted = sorted(all_features.keys())
		if not has_headers:
			header_line = "document," + ",".join(keys_sorted) + "\n"
			with open(feature_store_file, "a") as s:
				s.write(header_line)
			has_headers = True

		line = filename + ","
		for key in keys_sorted:
			line = line + str(all_features[key]) + ","
		line = line[0:-1] + "\n"
		with open(feature_store_file, "a") as s:
			s.write(line)
		n += 1
		if n > 100:
			break
	x = pandas.read_csv(filepath_or_buffer=feature_store_file)
