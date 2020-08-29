import nltk

from MSC_classifier import document_feature_csv_to_class_feature_csv
from MSC_features import FeatureExtractor


def ask_for_downloads():
	print("Do you need to download stuff?")
	nltk.download('punkt')
	nltk.download('averaged_perceptron_tagger')
	nltk.download('wordnet')
	nltk.download('stopwords')
	print("You downloaded stuff.")


def main():
	ask_for_downloads()
	fe = FeatureExtractor(static_features=["gold"])
	fe.list_file_to_feature_csv(list_file="lists/train_pos.txt", csv_file="a.csv", append=False,
								static_features={"gold": "pos"})
	fe.list_file_to_feature_csv(list_file="lists/train_neg.txt", csv_file="a.csv", append=True,
								static_features={"gold": "neg"})

	document_feature_csv_to_class_feature_csv(document_feature_csv_path="a.csv", class_label_column_name="gold",
											  label_feature_csv_path="b.csv")
	pass


if __name__ == '__main__':
	main()
