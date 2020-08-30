# MSC_classifier.py
# ..........
#
# Angelina-Sophia Hauswald
# Matrikel-Nr.: 785803
# OS: Ubuntu 20.04 LTS 
# Python 3.8.2 (default, Apr 27 2020, 15:53:34)
# [GCC 9.3.0] on linux

import pandas


# all_labels: 				Liste aller Labels, die es gibt
# class_label_column_name: 	Name der Spalte, in der pos/neg drin steht
# df:						eigentliches Dataframe
# return: dict, indem zu jedem Label (in dem Fall pos o. neg) 
# die durchschnittlichen Werte der errechneten Features stehen
def _label_averages_dict_from_dataframe(all_labels, class_label_column_name, df):
	# Liste von allen Features, die numerisch sind 
	numeric_features = _numeric_feature_keys_of_dataframe(df)
	if class_label_column_name in numeric_features:
		numeric_features.remove(class_label_column_name)
	#alphabetisch sortieren
	numeric_features = sorted(numeric_features)
	class_feature_averages_dict = {}
	for label in all_labels:
		class_feature_averages_dict[label] = {}
		# aus gesamten Dataframe gib mir ein dataframe, das nur diejenigen beinhaltet, dessen class label comumn name
		# gleich label
		class_df = df.loc[df[class_label_column_name] == label]
		# Durchschnitte berechnen, in dict speichern
		for feature in numeric_features:
			class_feature_averages_dict[label][feature] = class_df[feature].mean()
	return class_feature_averages_dict


# errechnet wieviele Dateien richtig neg und falsch pos klassifiziert wurden (und umgekehrt)
# prediction_csv: 	Dateiname, darin aufgelistet alle klassifizierten Dateien mit Prediction-Label
#					und tatsaechlichem Gold-Label
# retun: Matrix, in der steht, wieviele neg richtig neg und falsch klassifiziert wurden (und umgekehrt)
# Beispiel Matrix: {"neg": {"neg": 10, "pos": 20}, "pos":  {"neg": 30, "pos": 40}}
def read_confusion_matrix_from_prediction_csv(prediction_csv):
	#einlesen CSV
	x = pandas.read_csv(prediction_csv)
	labels = list(set(x["gold"].unique().tolist() + x["prediction"].unique().tolist()))
	confusion_matrix = {}
	for gold_label in labels:
		confusion_matrix[gold_label] = {}
		for predict_label in labels:
			confusion_matrix[gold_label][predict_label] = 0
	for index, row in x.iterrows():
		confusion_matrix[row["gold"]][row["prediction"]] = confusion_matrix[row["gold"]][row["prediction"]] + 1
	return confusion_matrix


# return: Liste von allen Featurenamen, die numerisch sind
def _numeric_feature_keys_of_dataframe(df):
	all_numeric_features = []
	for key in df.keys():
		if pandas.api.types.is_numeric_dtype(df[key]):
			all_numeric_features.append(key)
	return all_numeric_features


# schreibt in eine CSV-Datei für alle Klassen, die bekannt sind (in dem Fall pos, neg), die durchschnittlichen
# Werte, der gezaehlten Features
# document_feature_csv_path:	Dateipfad, der CSV, in der die Features zu den Trainingsdaten enthalten sind
# label_feature_csv_path:		Dateiname, der CSV, in der die durchschnittlichen Werte der Features eingeschrieben
#								werden sollen
# class_label_column_name:      
# only_numerical_features:		standardmaessig auf True, da die Formel nur Berechnung zu numerischen
#								Featuren erlaubt
def document_feature_csv_to_class_feature_csv(*, document_feature_csv_path, label_feature_csv_path,
											  class_label_column_name, only_numerical_features=True):
	if not only_numerical_features:
		raise NotImplementedError("Sorry, the given formula only allows for numerical features. ¯\_(ツ)_/¯")
	# einlesen der CSV
	df = pandas.read_csv(document_feature_csv_path)
	# gold, aus der spalte alle einzigartigen werte, also alle label die es gibt
	all_labels = df[class_label_column_name].unique().tolist()
	# dict, dass durchschnittliche Klassenpunkte speichert
	label_feature_averages_dict = _label_averages_dict_from_dataframe(all_labels, class_label_column_name, df)

	#schreiben in CSV
	with open(label_feature_csv_path, "w") as label_feature_csv:
		a_label = list(label_feature_averages_dict.keys())[0]
		features = sorted(label_feature_averages_dict[a_label].keys())
		label_feature_csv.write("label," + ",".join(features))
		for label in all_labels:
			line = label
			for feature in features:
				line += "," + str(label_feature_averages_dict[label][feature])
			label_feature_csv.write("\n" + line)


class Classifier:
	def __init__(self, csv_file, label_column_name="label", *, exclude_features=None, only_use_features=None) -> None:
		if exclude_features and only_use_features:
			raise Exception("You can only use either exclude_feature or only_use_features, not both.")
		# Liste aller Labels
		self._labels = []
		# Liste aller Features
		self._features = []
		# für jedes Label (in dem Fall neg/pos) die durchschnittlichen errechneten Werte der
		# Features gespeichert in einem
		# d.h. die Daten, der eingelesenen CSV, gespeichert in einem dict
		self._labels_feature_dict = {}
		# Features mit 1.0 gewichtet
		self._weights = {}
		# Dateiname
		self._csv_file = csv_file
		self.__load_from_csv(label_column_name, exclude_features=exclude_features, only_use_features=only_use_features)
		self.__set_weights()
		super().__init__()

	# schreibt in eine CSV den Dateinamen, das predicted Label zu der Datei und das Gold-Label
	# documents_csv_file:	Datei, die die Dateinamen und alle zugehoerigen berechneten Features enthaelt
	# target_csv_file: 		Dateiname der CSV, die erstellt wird
	# copy_fields: 			
	def classify_documents_in_file_and_save_to_csv(self, *, documents_csv_file, target_csv_file, copy_fields=["gold"]):
		# CSV in ein Dataframe einlesen
		df = pandas.read_csv(documents_csv_file)
		# konvertieren des Dataframes (also der CSV) in ein dict mit Dateinamen als Schluessel
		documents = df.set_index("filename").T.to_dict()
		#
		prediction_dictionaries = {} 
		for filename in documents.keys():
			prediction_dictionaries[filename] = {}
			# Bsp.: {"text1": {}}
			prediction_dictionaries[filename]["prediction"] = self.classify_document(documents[filename])
			# Bsp.: {"text1": {"prediction": "pos"}}
			for copy_field in copy_fields:
				prediction_dictionaries[filename][copy_field] = documents[filename][copy_field]
				# Bsp.: {"text1": {"prediction": "pos", "gold": "pos"}}

		# Dateiname, Prediction und alle anderen Werte, die kopiert werden sollen, in die target-CSV schreiben
		with open(target_csv_file, "w") as f:
			#Header-Zeile
			f.write("filename,prediction,")
			f.write(",".join(copy_fields))
			for filename in prediction_dictionaries.keys():
				line = filename + "," + prediction_dictionaries[filename]["prediction"]
				for copy_field in copy_fields:
					line += "," + prediction_dictionaries[filename][copy_field]
				f.write("\n" + line)


	# document: dict, in dem zu jeder Datei, die errechneten Features zu finden sind,
	# urspruenglich ausgelesen aus einer CSV-Datei
	# return: Label mit minimalstem Abstand (in dem Fall, entweder pos o. neg)
	def classify_document(self, document):
		distances = {}
		# für jedes Label Distanz berechnen
		for label in self._labels:
			label_dict = self._labels_feature_dict[label]
			distances[label] = self.__distance_between_documents(document, label_dict)
		# Liste mit allen minimalsten Distanzen (koennten auch mehrere sein, die minimalsten Abstand haben)
		min_value = min(distances.values())
		# wenn mehrere Label den minimalsten Abstand haben, wird einfach das erste Element dieser Liste ausgegeben
		predicted_labels = [key for key, value in distances.items() if value == min_value]
		return predicted_labels[0]

	# return: Distanz 
	def __distance_between_documents(self, doc_a, doc_b):
		distance = 0
		# durch alle Features durchgehen, die uns interessieren
		for feature in self._features:
			# Betrag von allen Differenzen aus den Feature-Paaren zweier Dokumente
			d = abs(doc_a[feature] - doc_b[feature])
			# Gewicht für jedes Feature auf multiplizieren
			d *= self._weights[feature]
			distance += d
		return distance

	# String-Repraesentation des Classifiers
	def __str__(self) -> str:
		return "Classifier from csv {} with labels {} and features {}".format(self._csv_file, self._labels,
																			  self._features)

	def __repr__(self) -> str:
		return str(self)


	# speichert in self._features alle Features, die extrahiert wurden
	# speichert in self._labels alle Labels, die Klassifizierer kennt (in dem Fall nur pos/neg)
	# speichert in self._labels_feature_dict zu jedem Label (in dem Fall nur pos/neg) die durchschnittlichen 
	# Werte der Features
	# label_column_name: 	Name des Klassenlabels (in dem Fall standardmaessig "Label")
	# exclude_features:		Features, die nicht verwendet werden sollen
	# only_use_features:	nur die Features, die verwendet werden sollen
	def __load_from_csv(self, label_column_name, *, exclude_features=None, only_use_features=None):
		# liest averages CSV ein (die die beiden label repraesentieren)
		df = pandas.read_csv(self._csv_file)
		# Liste aller Features, die extrahiert wurden
		self._features = self.__collect_wanted_features_from_dataframe(df=df, label_column_name=label_column_name,
																	   exclude_features=exclude_features,
																	   only_use_features=only_use_features)
		# Liste aller Labels, die der Klassifier kennt (in dem Fall nur pos, neg)
		self._labels = list(df[label_column_name].unique())
		# Dataframe nicht mit einer Zeilennummer, sondern mit Namen des Labels adressieren
		df = df.set_index(label_column_name)
		# für jedes Label, Eintrag in eigenes dict mit den Werten, die uns interessieren
		# d.h. die die angegeben sind in exclude_features oder only_use_features
		for label in self._labels:
			self._labels_feature_dict[label] = {}
			for feature in self._features:
				self._labels_feature_dict[label][feature] = df.T[label][feature]


	# liest aus dem df die Featurenamen aus, die uns interessieren und gibt diese zurueck
	# df: Dataframe, der eingelesenen CSV-Datei
	# label_column_name: 	Name des Klassenlabels (in dem Fall standardmaessig "Label")
	# exclude_features:		Features, die nicht verwendet werden sollen
	# only_use_features:	nur die Features, die verwendet werden sollen
	# return: Liste von Namen der Features, die uns interessieren
	def __collect_wanted_features_from_dataframe(self, *, df, label_column_name, exclude_features, only_use_features):
		csv_features = list(df.keys())
		# Label raus schmeißen, weil es kein feature ist
		csv_features.remove(label_column_name)
		# wenn es features gibt, die excludiert wreden sollen, dann werden diese nicht hinzugefuegt
		if exclude_features:
			for exclude_feature in exclude_features:
				csv_features.remove(exclude_feature)
		# wenn es features gibt, die ausschliesslich beutzt werden sollen
		if only_use_features:
			# wenn gewolltes Feature nicht in CSV entnhalten ist, dann Fehlermeldung
			for wanted_featue in only_use_features:
				if wanted_featue not in csv_features:
					raise Exception(
						"The explicitly expected feature {} is not listed in the csv file {}.".format(wanted_featue,
																									  self._csv_file))
			# nur die Features zurueck geben, die gewollt sind
			csv_features = only_use_features
		return csv_features

	# setzt Gewichte aller Features auf 1.0
	def __set_weights(self):
		for feature in self._features:
			self._weights[feature] = 1.0
