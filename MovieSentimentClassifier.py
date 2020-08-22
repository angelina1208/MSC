import os

import nltk

from MSC_features import regex_countable_features, nltk_countable_features
import pandas

def ask_for_downloads():
	print("Do you need to download stuff?")
	nltk.download('punkt')
	nltk.download('averaged_perceptron_tagger')
	nltk.download('wordnet')
	print("You downloaded stuff.")


def main():
	print("Welcome to MSC.")
	ask_for_downloads()
	pos_folder = "aclImdb_v1/aclImdb/train/pos"
	neg_folder = "aclImdb_v1/aclImdb/train/neg"
	data_features_pos = "feats_pos.csv"
	data_features_neg = "feats_neg.csv"
	data_features_summary = "feats_summary.csv"
	print("Training with positive samples from: {}\nand negative samples from: {}".format(pos_folder, neg_folder))

	# extract_features_and_store_in_csv(data_features_pos, pos_folder, "pos")
	# extract_features_and_store_in_csv(data_features_neg, neg_folder, "neg")

	negg = pandas.read_csv(data_features_neg)
	poss = pandas.read_csv(data_features_pos)
	sorted_keys = sorted(negg.keys())
	negg_line = "neg"
	poss_line = "pos"
	feature_keys = ",".join(sorted_keys)
	for key in sorted_keys:
		if pandas.api.types.is_numeric_dtype(negg[key]):
			negg_line += "," + str(negg[key].mean())
			poss_line += "," + str(poss[key].mean())
	with open(data_features_summary, "w") as f:
		f.write("label," + feature_keys)
		f.write("\n")
		f.write(negg_line)
		f.write("\n")
		f.write(poss_line)
		f.write("\n")




def extract_features_and_store_in_csv(feature_store_file, folder, goldlabel, should_delete_existing_csv=True):
	if should_delete_existing_csv and os.path.isfile(feature_store_file):
		os.remove(feature_store_file)
	has_headers = False
	filenames = list(os.listdir(folder))
	print("Found {} files".format(len(filenames)))
	n = 0
	for filename in filenames:
		with open(os.path.join(os.getcwd(), folder, filename), 'r') as f:
			text = f.read()

		all_features = {}
		all_features.update(regex_countable_features(text))
		all_features.update(nltk_countable_features(text))
		keys_sorted = sorted(all_features.keys())
		if not has_headers:
			header_line = "document,goldlabel," + ",".join(keys_sorted) + "\n"
			with open(feature_store_file, "a") as s:
				s.write(header_line)
			has_headers = True

		line = filename + "," + goldlabel + ","
		for key in keys_sorted:
			line = line + str(all_features[key]) + ","
		line = line[0:-1] + "\n"
		with open(feature_store_file, "a") as s:
			s.write(line)
		n += 1
		if n % 100 == 0:
			print("Read {} files".format(n))


if __name__ == '__main__':
	main()
