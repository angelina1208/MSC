import os

import nltk
import pandas

from MSC_features import regex_countable_features, nltk_countable_features


def ask_for_downloads():
	print("Do you need to download stuff?")
	nltk.download('punkt')
	nltk.download('averaged_perceptron_tagger')
	nltk.download('wordnet')
	print("You downloaded stuff.")


def classify_documents_in_csv_to_dict(classifier, source_csv):
	source_dataframe = pandas.read_csv(source_csv)
	documents = source_dataframe.set_index("filename").T.to_dict()
	diccts = {}
	for filename in documents.keys():
		diccts[filename] = {}
		diccts[filename]["prediction"] = classifier.classify(documents[filename])
	return diccts


def main():
	print("Welcome to MSC.")
	ask_for_downloads()
	train_pos_files_list = "train_pos.txt"
	train_neg_files_list = "train_neg.txt"
	validate_pos_files_list = "validate_pos.txt"
	validate_neg_files_list = "validate_neg.txt"
	test_pos_files_list = "test_pos.txt"
	test_neg_files_list = "test_neg.txt"
	data_features_pos = "feats_pos.csv"
	data_features_neg = "feats_neg.csv"
	data_features_summary = "feats_summary.csv"

	print("Training with positive samples from: {}\nand negative samples from: {}".format(train_pos_files_list,
																						  train_neg_files_list))
	train(data_features_neg, data_features_pos, data_features_summary, train_neg_files_list, train_pos_files_list)

	for_all_files_in_list_extract_features_and_store_in_file(validate_neg_files_list, "validate_neg.csv")
	for_all_files_in_list_extract_features_and_store_in_file(validate_pos_files_list, "validate_pos.csv")

	run_validation(data_features_summary)

	evaluate_prediction("x.csv")

def evaluate_prediction(prediction_csv):
	x = pandas.read_csv(prediction_csv)
	labels = list(set(x["gold"].unique().tolist() + x["prediction"].unique().tolist()))
	print(labels)
	confusion = {}
	for gold_label in labels:
		confusion[gold_label] = {}
		for predict_label in labels:
			confusion[gold_label][predict_label] = 0
	for index, row in x.iterrows():
		confusion[row["gold"]][row["prediction"]] = confusion[row["gold"]][row["prediction"]] + 1
	all = 0
	correct = 0
	for gold_label in labels:
		correct += confusion[gold_label][gold_label]
		for predict_label in labels:
			all += confusion[gold_label][predict_label]
	print(confusion)
	print(all)
	print(correct)
	print(correct / all)


def run_validation(data_features_summary):
	c = Classifier(data_features_summary)
	c.load()
	validate_neg_dict = classify_documents_in_csv_to_dict(c, "validate_neg.csv")
	validate_pos_dict = classify_documents_in_csv_to_dict(c, "validate_pos.csv")
	for key in validate_neg_dict.keys():
		validate_neg_dict[key]["gold"] = "neg"
	for key in validate_pos_dict.keys():
		validate_pos_dict[key]["gold"] = "pos"
	validate_predictions = {}
	validate_predictions.update(validate_neg_dict)
	validate_predictions.update(validate_pos_dict)
	with open("x.csv", "w") as f:
		f.write("filename,gold,prediction\n")
	for key in validate_predictions.keys():
		line = key + "," + validate_predictions[key]["gold"] + "," + validate_predictions[key]["prediction"] + "\n"
		with open("x.csv", "a") as f:
			f.write(line)


class Classifier:

	def __init__(self, class_csv) -> None:
		self.class_csv = class_csv
		self.label_dictionaries = None
		self.features = None
		self.weights = None
		super().__init__()

	def load(self):
		x = pandas.read_csv(self.class_csv)
		self.label_dictionaries = x.set_index("label").T.to_dict()
		self.features = [feature for feature in x.keys().to_list() if feature != "label"]
		self.weights = {}
		for feature in self.features:
			self.weights[feature] = 1.0

	def classify(self, document):
		distances = {}
		for label in self.label_dictionaries.keys():
			distance = 0
			for feature in self.features:
				d = abs(document[feature] - self.label_dictionaries[label][feature])
				d *= self.weights[feature]
				distance += d
			distances[label] = distance

		min_value = min(distances.values())
		predicted_labels = [key for key, value in distances.items() if value == min_value]
		return predicted_labels[0]

	def __str__(self) -> str:
		return super().__str__()

	def __repr__(self) -> str:
		return super().__repr__()


def train(data_features_neg, data_features_pos, data_features_summary, train_neg_files_list, train_pos_files_list):
	for_all_files_in_list_extract_features_and_store_in_file(train_pos_files_list, data_features_pos,
															 goldlabel="pos")
	for_all_files_in_list_extract_features_and_store_in_file(train_neg_files_list, data_features_neg,
															 goldlabel="neg")
	TODO(data_features_neg, data_features_pos, data_features_summary)


def TODO(data_features_neg, data_features_pos, data_features_summary):
	negg = pandas.read_csv(data_features_neg)
	poss = pandas.read_csv(data_features_pos)
	sorted_keys = sorted(negg.keys())
	negg_line = "neg"
	poss_line = "pos"
	header_line = "label"
	for key in sorted_keys:
		if pandas.api.types.is_numeric_dtype(negg[key]):
			negg_line += "," + str(negg[key].mean())
			poss_line += "," + str(poss[key].mean())
			header_line += "," + key
	with open(data_features_summary, "w") as f:
		f.write(header_line)
		f.write("\n")
		f.write(negg_line)
		f.write("\n")
		f.write(poss_line)
		f.write("\n")


def for_all_files_in_list_extract_features_and_store_in_file(folder, feature_store_file, *, goldlabel=None,
															 start_new_csv=True):
	all_features = feature_dict_for_text("")
	keys_sorted = sorted(all_features.keys())

	if start_new_csv:
		if os.path.isfile(feature_store_file):
			os.remove(feature_store_file)
		header_line = "filename," + ",".join(keys_sorted) + "\n"
		with open(feature_store_file, "a") as s:
			s.write(header_line)

	with open(folder, "r") as list_file:
		files = list_file.readlines()
	# Entferne zeilenumbrueche aus eintraegen
	files = [x.strip() for x in files]
	print("Found {} files".format(len(files)))

	for filename in files:
		with open(filename, 'r') as f:
			text = f.read()

		all_features = feature_dict_for_text(text)

		if goldlabel:
			all_features.update({"goldlabel": goldlabel})

		line = dict_and_filename_to_csv_line(all_features, filename)

		with open(feature_store_file, "a") as s:
			s.write(line)


def dict_and_filename_to_csv_line(all_features, filename):
	keys_sorted = sorted(all_features.keys())
	line = filename + ","
	for key in keys_sorted:
		line = line + str(all_features[key]) + ","
	line = line[0:-1] + "\n"
	return line


def feature_dict_for_text(text):
	all_features = {}
	all_features.update(regex_countable_features(text))
	all_features.update(nltk_countable_features(text))
	return all_features


if __name__ == '__main__':
	main()
