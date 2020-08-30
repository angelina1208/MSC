# install.py
# da eventuell ein paar NLTK Lexika installiert werden muessen,
# werden diese in diesem Skript bereitgestellt
#
# Angelina-Sophia Hauswald
# Matrikel-Nr.: 785803
# OS: Ubuntu 20.04 LTS \n \l
# Python 3.8.2 (default, Apr 27 2020, 15:53:34)
# [GCC 9.3.0] on linux

import nltk

if __name__ == '__main__':
	print("Some nltk lexicons must be donwloaded")
	nltk.download('punkt')
	nltk.download('averaged_perceptron_tagger')
	nltk.download('wordnet')
	nltk.download('stopwords')
	nltk.download('vader_lexicon')
	print("Downloads complete")