# example.py
# demonstriert die Benutzung und Arbeitsweise des Movie Sentiment Classifier
#
# Angelina-Sophia Hauswald
# Matrikel-Nr.: 785803
# OS: Ubuntu 20.04 LTS 
# Python 3.8.2 


from src.MSC_classifier import Classifier, read_confusion_matrix_from_prediction_csv, document_feature_csv_to_class_feature_csv
from src.MSC_features import FeatureExtractor
from os import mkdir
from os.path import isdir

def main():
	# wenn ueber die Features hinaus, die im Code festgelegt sind, welche extrahiert werden sollen, 
	# noch weitere features gebraucht werden, dann muessen diese dem Konstruktor gegeben werden
	# standardmaessig nur gold, fuer den Goldstandard, dieser ist ja in der Ordnerstruktur des Datensatzes zu finden
	# ob es sich um pos o. neg Daten handelt wird den jeweiligen Funktionen uebergeben, damit der Goldstandard
	# auch abgespeichert werden kann
	fe = FeatureExtractor(static_features=["gold"])

	# Erstellt das data directory falls noetig
	# in dem alle csvs stehen sollen
	if not isdir("data"):
		mkdir("data")

	# TRAINING:
	# extrahieren aller Features aus den positiven Trainingsdaten und speichert diese in data/train.csv
	# list_file: Dateipfad, in denen sich Daten befinden, aus denen Features extrahiert werden sollen
	# csv_file: Datei, in der Ergebnisse der Feature Extraktion gespeichert werden
	# append: faslse, wenn neue CSV erstellt werden soll, true, wenn CSV bereits existiert und nur noch
	#			in die bereits vorhandene geschrieben werden soll
	# statict_features: Wert für den Goldstandard, ob es sich um pos o. neg daten handelt
	fe.list_file_to_feature_csv(list_file="lists/train_pos.txt", csv_file="data/train.csv", append=False,
								static_features={"gold": "pos"})
	# extrahieren aller Features aus den negativen Trainingsdaten und speichert diese in data/train.csv
	fe.list_file_to_feature_csv(list_file="lists/train_neg.txt", csv_file="data/train.csv", append=True,
								static_features={"gold": "neg"})
	
	# schreibt in eine CSV-Datei für alle Klassen, die bekannt sind (in dem Fall pos, neg), die durchschnittlichen
	# Werte, der gezaehlten Features
	# document_feature_csv_path:	Dateipfad, der CSV, in der die Features zu den Trainingsdaten enthalten sind
	# label_feature_csv_path:		Dateiname, der CSV, in der die durchschnittlichen Werte der Features eingeschrieben
	#								werden sollen
	# class_label_column_name:      standardmaessig auf "gold", für den Goldstandard		
	document_feature_csv_to_class_feature_csv(document_feature_csv_path="data/train.csv", class_label_column_name="gold",
											  label_feature_csv_path="data/averages.csv")
	
	# VALIDIERUNG:
	# data to classify to csv
	# extrahieren aller Features aus den positiven und negative Trainingsdaten und speichern in data/train.csv
	fe.list_file_to_feature_csv(list_file="lists/validation_neg.txt", csv_file="data/validation.csv", append=False,
								static_features={"gold": "neg"})
	fe.list_file_to_feature_csv(list_file="lists/validation_pos.txt", csv_file="data/validation.csv", append=True,
								static_features={"gold": "pos"})
	
	
	# Einlesen des Klassifizierers von der CSV-Datei
	# Erstellen eines Objekts
	c = Classifier("data/averages.csv", only_use_features=["vader_compound", "ratio_char_questionmark"])
	# erstellen einer neuen prediction-csv, in der zu jeder Datei die prediction und der erwartete Goldstandard steht
	# documents_csv_file:	Datei, die die Dateinamen und alle zugehoerigen berechneten Features enthaelt
	# target_csv_file: 		Dateiname der CSV, die erstellt wird
	c.classify_documents_in_file_and_save_to_csv(documents_csv_file="data/validation.csv", target_csv_file="data/validation_prediction.csv")
	
	# Uberpruefen wie gut predictions sind 
	confusion_matrix = read_confusion_matrix_from_prediction_csv("data/validation_prediction.csv")
	# Ergebnisse praesentieren
	print_confusion_matrix_with_all_stats(confusion_matrix)
	
	
	# TESTEN:
	# Features extrahieren
	fe.list_file_to_feature_csv(list_file="lists/test_neg.txt", csv_file="data/test.csv", append=False,
								static_features={"gold": "neg"})
	fe.list_file_to_feature_csv(list_file="lists/test_pos.txt", csv_file="data/test.csv", append=True,
							static_features={"gold": "pos"})

	# Dokumente klassifizieren
	c.classify_documents_in_file_and_save_to_csv(documents_csv_file="data/test.csv", target_csv_file="data/test_prediction.csv")

	# Ueberpruefen, wie gut predictions sind
	confusion_matrix = read_confusion_matrix_from_prediction_csv("data/test_prediction.csv")
	# Ergebnisse praesentieren
	print_confusion_matrix_with_all_stats(confusion_matrix)

# gibt die Confusion-Matrix und die Fehlerrrate und Erkennungsrate aus
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