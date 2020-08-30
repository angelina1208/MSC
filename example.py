# example.py
# demonstriert den Movie Sentiment Classifier
#
# Angelina-Sophia Hauswald
# Matrikel-Nr.: 785803
# OS: Ubuntu 20.04 LTS 
# Python 3.8.2 (default, Apr 27 2020, 15:53:34)
# [GCC 9.3.0] on linux

from src.MSC_classifier import Classifier, read_confusion_matrix_from_prediction_csv, document_feature_csv_to_class_feature_csv
from src.MSC_features import FeatureExtractor
from os import mkdir
from os.path import isdir

def main():
	fe = FeatureExtractor(static_features=["gold"])

	# Erstellt das data directory falls noetig
	# in dem alle csvs stehen sollen
	if not isdir("data"):
		mkdir("data")

	# TRAIN
	# Train files to one data/train.csv
	# extrahieren aller Features aus den positiven Trainingsdaten
	fe.list_file_to_feature_csv(list_file="lists/train_pos.txt", csv_file="data/train.csv", append=False,
								static_features={"gold": "pos"})
	# extrahieren aller Features aus den negativen Trainingsdaten
	fe.list_file_to_feature_csv(list_file="lists/train_neg.txt", csv_file="data/train.csv", append=True,
								static_features={"gold": "neg"})
	
	# average data/train.csv
	document_feature_csv_to_class_feature_csv(document_feature_csv_path="data/train.csv", class_label_column_name="gold",
											  label_feature_csv_path="data/averages.csv")
	
	# VALIDATE
	# data to classify to csv
	fe.list_file_to_feature_csv(list_file="lists/validation_neg.txt", csv_file="data/validation.csv", append=False,
								static_features={"gold": "neg"})
	fe.list_file_to_feature_csv(list_file="lists/validation_pos.txt", csv_file="data/validation.csv", append=True,
								static_features={"gold": "pos"})
	
	#read the classifier from csv
	# erstellen eines Obejkts
	c = Classifier("data/averages.csv", only_use_features=["vader_compound", "ratio_char_questionmark"])
	#create new predicitons csv
	c.classify_documents_in_file_and_save_to_csv(documents_csv_file="data/validation.csv", target_csv_file="data/validation_prediction.csv")
	
	# check how good predtions are
	confusion_matrix = read_confusion_matrix_from_prediction_csv("data/validation_prediction.csv")
	print_confusion_matrix_with_all_stats(confusion_matrix)
	
	
	# TEST
	fe.list_file_to_feature_csv(list_file="lists/test_neg.txt", csv_file="data/test.csv", append=False,
								static_features={"gold": "neg"})
	fe.list_file_to_feature_csv(list_file="lists/test_pos.txt", csv_file="data/test.csv", append=True,
							static_features={"gold": "pos"})
	
	c.classify_documents_in_file_and_save_to_csv(documents_csv_file="data/test.csv", target_csv_file="data/test_prediction.csv")

	confusion_matrix = read_confusion_matrix_from_prediction_csv("data/test_prediction.csv")
	print_confusion_matrix_with_all_stats(confusion_matrix)

# Die funktion schreibt die confusion matrix und ausserdem
# die fehlerrrate und erkennungsrate
def print_confusion_matrix_with_all_stats(confusion_matrix):
	print(confusion_matrix)
	num_all = 0
	num_correct = 0
	for expected in confusion_matrix:
		for predicted in confusion_matrix[expected]:
			num_all += confusion_matrix[expected][predicted]
			if expected == predicted:
				num_correct += confusion_matrix[expected][predicted]
	print("Documents:{}".format(num_all))
	print("Correct:{}".format(num_correct))
	print("Prediction_rate:{}".format(num_correct / num_all))
	print("Error_rate:{}".format(1 - (num_correct / num_all)))


if __name__ == '__main__':
	main()



#wenn ueber die features die er errechnet noch weitere features braucht,
	#dann muss das beim konstruktor angegeben werden
	#es gibt noch ne zeile gold ['gold']