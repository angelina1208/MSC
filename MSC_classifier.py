import pandas


def document_feature_csv_to_class_feature_csv(*, document_feature_csv_path, label_feature_csv_path,
											  class_label_column_name, only_numerical_features=True):
	if not only_numerical_features:
		raise NotImplementedError("Sorry, the given formula only allows for numerical features. ¯\_(ツ)_/¯")

	df = pandas.read_csv(document_feature_csv_path)

	all_labels = df[class_label_column_name].unique().tolist()
	label_feature_averages_dict = _label_averages_dict(all_labels, class_label_column_name, df)

	with open(label_feature_csv_path, "w") as label_feature_csv:
		a_label = list(label_feature_averages_dict.keys())[0]
		features = sorted(label_feature_averages_dict[a_label].keys())
		label_feature_csv.write("label," + ",".join(features))
		for label in all_labels:
			line = label
			for feature in features:
				line += "," + str(label_feature_averages_dict[label][feature])
			label_feature_csv.write("\n" + line)


def _label_averages_dict(all_labels, class_label_column_name, df):
	numeric_features = _numeric_feature_keys_of_dataframe(df)
	if class_label_column_name in numeric_features:
		numeric_features.remove(class_label_column_name)
	numeric_features = sorted(numeric_features)
	class_feature_averages_dict = {}
	for label in all_labels:
		class_feature_averages_dict[label] = {}
		class_df = df.loc[df[class_label_column_name] == label]
		for feature in numeric_features:
			class_feature_averages_dict[label][feature] = class_df[feature].mean()
	return class_feature_averages_dict


def _numeric_feature_keys_of_dataframe(df):
	all_numeric_features = []
	for key in df.keys():
		if pandas.api.types.is_numeric_dtype(df[key]):
			all_numeric_features.append(key)
	return all_numeric_features


class Classifier:
	def __init__(self, csv_file, label_column_name="label", *, exclude_features=None, only_use_features=None) -> None:
		if exclude_features and only_use_features:
			raise Exception("You can only use either exclude_feature or only_use_features, not both.")
		self._labels = []
		self._features = []
		self._labels_feature_dict = {}
		self._weights = {}
		self._csv_file = csv_file
		self._load_from_csv(label_column_name, exclude_features=exclude_features, only_use_features=only_use_features)
		self._set_weights()
		super().__init__()

	def classify_documents_in_file_and_save_to_csv(self, *, documents_csv_file, target_csv_file, copy_fields=["gold"]):
		df = pandas.read_csv(documents_csv_file)
		documents = df.set_index("filename").T.to_dict()
		prediction_dictionaries = {}
		for filename in documents.keys():
			prediction_dictionaries[filename] = {}
			prediction_dictionaries[filename]["prediction"] = self.classify_document(documents[filename])
			for copy_field in copy_fields:
				prediction_dictionaries[filename][copy_field] = documents[filename][copy_field]

		with open(target_csv_file, "w") as f:
			f.write("filename,prediction,")
			f.write(",".join(copy_fields))
			for filename in prediction_dictionaries.keys():
				line = filename + "," + prediction_dictionaries[filename]["prediction"]
				for copy_field in copy_fields:
					line+= "," + prediction_dictionaries[filename][copy_field]
				f.write("\n"+line)

	def classify_document(self, document):
		distances = {}
		for label in self._labels:
			label_dict = self._labels_feature_dict[label]
			distances[label] = self._distance_between_documents(document, label_dict)
		min_value = min(distances.values())
		predicted_labels = [key for key, value in distances.items() if value == min_value]
		return predicted_labels[0]

	def _distance_between_documents(self, doc_a, doc_b):
		distance = 0
		for feature in self._features:
			d = abs(doc_a[feature] - doc_b[feature])
			d *= self._weights[feature]
			distance += d
		return distance

	def __str__(self) -> str:
		return "Classifier from csv {} with labels {} and features {}".format(self._csv_file, self._labels,
																			  self._features)

	def __repr__(self) -> str:
		return str(self)

	def _load_from_csv(self, label_column_name, *, exclude_features=None, only_use_features=None):
		df = pandas.read_csv(self._csv_file)
		self._features = self._collect_wanted_features_from_dataframe(df=df, label_column_name=label_column_name,
																	  exclude_features=exclude_features,
																	  only_use_features=only_use_features)

		self._labels = list(df[label_column_name].unique())
		df = df.set_index(label_column_name)
		for label in self._labels:
			self._labels_feature_dict[label] = {}
			for feature in self._features:
				self._labels_feature_dict[label][feature] = df.T[label][feature]

	def _collect_wanted_features_from_dataframe(self, *, df, label_column_name, exclude_features, only_use_features):
		csv_features = list(df.keys())
		csv_features.remove(label_column_name)
		if exclude_features:
			for exclude_feature in exclude_features:
				csv_features.remove(exclude_feature)
		if only_use_features:
			for wanted_featue in only_use_features:
				if wanted_featue not in only_use_features:
					raise Exception(
						"The explicitly expected feature {} is not listed in the csv file {}.".format(wanted_featue,
																									  self._csv_file))
			csv_features = only_use_features
		return csv_features

	def _set_weights(self):
		for feature in self._features:
			self._weights[feature] = 1.0


def read_confusion_matrix_from_prediction_csv(prediction_csv):
	x = pandas.read_csv(prediction_csv)
	labels = list(set(x["gold"].unique().tolist() + x["prediction"].unique().tolist()))
	print(labels)
	confusion_matrix = {}
	for gold_label in labels:
		confusion_matrix[gold_label] = {}
		for predict_label in labels:
			confusion_matrix[gold_label][predict_label] = 0
	for index, row in x.iterrows():
		confusion_matrix[row["gold"]][row["prediction"]] = confusion_matrix[row["gold"]][row["prediction"]] + 1
	return confusion_matrix