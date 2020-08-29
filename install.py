import nltk

if __name__ == '__main__':
	print("Some nltk lexicons must be donwloaded")
	nltk.download('punkt')
	nltk.download('averaged_perceptron_tagger')
	nltk.download('wordnet')
	nltk.download('stopwords')
	nltk.download('vader_lexicon')
	print("Downloads complete")