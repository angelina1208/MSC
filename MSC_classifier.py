import pandas


def document_feature_csv_to_class_feature_csv(*, document_feature_csv_path, label_feature_csv_path,
											  class_label_column_name, only_numerical_features=True):
	if not only_numerical_features:
		raise NotImplementedError("Sorry, the given formula only allows for numerical features. ¯\_(ツ)_/¯")

	df = pandas.read_csv(document_feature_csv_path)

	all_labels = df[class_label_column_name].unique().tolist()
	label_feature_averages_dict = label_averages_dict(all_labels, class_label_column_name, df)

	with open(label_feature_csv_path, "w") as label_feature_csv:
		a_label = list(label_feature_averages_dict.keys())[0]
		features = sorted(label_feature_averages_dict[a_label].keys())
		label_feature_csv.write("label," + ",".join(features))
		for label in all_labels:
			line = label
			for feature in features:
				line += "," + str(label_feature_averages_dict[label][feature])
			label_feature_csv.write("\n" + line)


def label_averages_dict(all_labels, class_label_column_name, df):
	numeric_features = _numeric_feature_keys_of_dataframe(df)
	if class_label_column_name in numeric_features:
		numeric_features.remove(class_label_column_name)
	numeric_features = sorted(numeric_features)
	class_feature_averages_dict = {}
	for label in all_labels:
		class_feature_averages_dict[label] = {}
		class_df = df.loc[df[class_label_column_name] == label]
		for feature in numeric_features:
			class_feature_averages_dict[label]["avg_" + feature] = class_df[feature].mean()
	return class_feature_averages_dict


def _numeric_feature_keys_of_dataframe(df):
	all_numeric_features = []
	for key in df.keys():
		if pandas.api.types.is_numeric_dtype(df[key]):
			all_numeric_features.append(key)
			nlt
	return all_numeric_features


class Classifier:
	pass