#!/usr/bin/python

import re
import os

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from util import isValidWord

LINK_REG = re.compile(r'\[ ([^[\]()]*) \] \( ([^[\]()]*) \)')
URL_REG = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

def preprocess(text):
	# replace links of form: [ text ] ( url ) with just text
	text = LINK_REG.sub(r'\1', text)

	# replace urls with URL
	text = URL_REG.sub(r'<URL>', text)

	return text

def write_corpus(infile, outfile, newline='\n'):
	f = open(infile, 'r')
	g = open(outfile, 'w')
	i = 0

	for line in f:
		if i+1 % 100000 == 0:
			print(i)

		match = DATA_REG.match(line.strip())
		if match is None:
			text = preprocess(line)
		else:
			user, sub, text = match.group(1, 2, 3)
			text = preprocess(text)

		g.write(text.strip().lower())
		g.write(newline)
		i += 1

if __name__ == "__main__":
	for file in os.listdir("./reddit.l2"):

		FILE = 'reddit.l2/' + file
		OUTFILE = 'corpus/' + file + ".cleanedforw2v"
		W2V_OUTFILE = OUTFILE + ".w2v"
		
		if(not os.path.isfile(W2V_OUTFILE) and not os.path.isfile(OUTFILE)):

			print "Constructing word vectors for " + str(file)
			print "Parsing Training Data"

			write_corpus(FILE, OUTFILE, newline="\n")
			sentences = LineSentence(OUTFILE)

			print "Training w2v model... This may take a while"
			model = Word2Vec(sentences=sentences, size=50, max_final_vocab=50000, workers=10, iter=3)

			print "Training Complete... Got", len(model.wv.vocab), "unique word vectors"

			print "Saving vectors to disk"
			f = open(W2V_OUTFILE, "w")
			for word in model.wv.vocab:
				if(isValidWord(word)):
					try:
						f.write(word + " " + " ".join([str(float(x)) for x in model.wv[word]]) + "\n")
					except UnicodeEncodeError as e:
						pass
			f.close()
		else:
			print "Passing on", FILE, "because we found corresponding cleaned and w2v files"
