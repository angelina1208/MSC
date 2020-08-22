import os

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
	for filename in os.listdir(pos_folder):
		with open(os.path.join(os.getcwd(), pos_folder, filename), 'r') as f:
			text = f.read()

		# print(text)
		all_features = {}
		all_features.update(regex_countable_features(text))
		all_features.update(nltk_countable_features(text))
		line = filename + ","
		keys_sorted = sorted(all_features.keys())
		for key in keys_sorted:
			line = line + str(all_features[key]) + ","
		line = line[0:-1] + "\n"
		with open(feature_store_file, "a") as s:
			s.write(line)



# print(all_features)
