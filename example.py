import nltk

from MSC_classifier import Classifier, read_confusion_matrix_from_prediction_csv, document_feature_csv_to_class_feature_csv
from MSC_features import FeatureExtractor

def main():
	# fe = FeatureExtractor(static_features=["gold"])
	#
	# #train
	#
	# # Train files to one train csv
	#
	# fe.list_file_to_feature_csv(list_file="lists/train_pos.txt", csv_file="train.csv", append=False,
	# 							static_features={"gold": "pos"})
	# fe.list_file_to_feature_csv(list_file="lists/train_neg.txt", csv_file="train.csv", append=True,
	# 							static_features={"gold": "neg"})
	#
	# # average train csv
	# document_feature_csv_to_class_feature_csv(document_feature_csv_path="train.csv", class_label_column_name="gold",
	# 										  label_feature_csv_path="classifier.csv")
	#
	# #validate
	#
	# # data to classify to csv
	# fe.list_file_to_feature_csv(list_file="lists/validate_neg.txt", csv_file="validation.csv", append=False,
	# 							static_features={"gold": "neg"})
	# fe.list_file_to_feature_csv(list_file="lists/validate_pos.txt", csv_file="validation.csv", append=True,
	# 							static_features={"gold": "pos"})
	#
	# #read the classifier from csv
	# c = Classifier("classifier.csv", only_use_features=["vader_compound", "ratio_char_questionmark"])
	# #create new predicitons csv
	# c.classify_documents_in_file_and_save_to_csv(documents_csv_file="validation.csv", target_csv_file="validation_prediction.csv")
	#
	# # check how good predtions are
	# confusion_matrix = read_confusion_matrix_from_prediction_csv("validation_prediction.csv")
	# print(confusion_matrix)
	#
	#
	# #test
	#
	# fe.list_file_to_feature_csv(list_file="lists/test_neg.txt", csv_file="test.csv", append=False,
	# 							static_features={"gold": "neg"})
	# fe.list_file_to_feature_csv(list_file="lists/test_pos.txt", csv_file="test.csv", append=True,
	# 						static_features={"gold": "pos"})
	#
	# c.classify_documents_in_file_and_save_to_csv(documents_csv_file="test.csv", target_csv_file="test_prediction.csv")

	confusion_matrix = read_confusion_matrix_from_prediction_csv("test_prediction.csv")
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
	print("Prediction_rate:{}".format(num_correct/num_all))
	print("Error_rate:{}".format(1 - (num_correct/num_all)))

if __name__ == '__main__':
	main()
