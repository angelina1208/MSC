import nltk

from MSC_classifier import Classifier, read_confusion_matrix_from_prediction_csv, document_feature_csv_to_class_feature_csv
from MSC_features import FeatureExtractor

def main():
	# fe = FeatureExtractor(static_features=["gold"])
	# fe.list_file_to_feature_csv(list_file="lists/train_pos.txt", csv_file="a.csv", append=False,
	# 							static_features={"gold": "pos"})
	# fe.list_file_to_feature_csv(list_file="lists/train_neg.txt", csv_file="a.csv", append=True,
	# 							static_features={"gold": "neg"})
	#
	# document_feature_csv_to_class_feature_csv(document_feature_csv_path="a.csv", class_label_column_name="gold",
	# 										  label_feature_csv_path="b.csv")
	#
	# fe.list_file_to_feature_csv(list_file="lists/validate_neg.txt", csv_file="aa.csv", append=False,
	# 							static_features={"gold": "neg"})
	# fe.list_file_to_feature_csv(list_file="lists/validate_pos.txt", csv_file="aa.csv", append=True,
	# 							static_features={"gold": "pos"})


	c = Classifier("b.csv")
	c = Classifier("b.csv", only_use_features=["vader_compound", "ratio_char_questionmark"])
	c.classify_documents_in_file_and_save_to_csv(documents_csv_file="aa.csv", target_csv_file="ab.csv")
	read_confusion_matrix_from_prediction_csv("ab.csv")


if __name__ == '__main__':
	main()
