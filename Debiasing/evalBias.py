from matplotlib import pylab
from scipy import spatial
from pyvttbl import Anova1way
import numpy as np
import itertools

def generateAnalogies(analogyTemplates, keyedVecs):
	expandedAnalogyTemplates = []
	for A, stereotypes in analogyTemplates.items():
		for B, _ in analogyTemplates.items():
			if(A != B):
				for stereotype in stereotypes:
					expandedAnalogyTemplates.append([[B, stereotype], [A]])

	analogies = []
	for positive, negative in expandedAnalogyTemplates:
		words = []
		try:
			words = keyedVecs.most_similar(positive=positive, negative=negative, topn=100)
		except KeyError as e:
			pass
			
		for word, score in words:
			analogy = str(negative[0]) + " is to " + str(positive[1]) + " as " + str(positive[0]) + " is to " + str(word)
			analogyRaw = [negative[0], positive[1], positive[0], word]
			analogies.append([score, analogy, analogyRaw])

	analogies = sorted(analogies, key=lambda x:-x[0])
	return analogies

def multiclass_evaluation(embeddings, targets, attributes):
	targets_eval = []
	for targetSet in targets:
		for target in targetSet:
			for attributeSet in attributes:
				targets_eval.append(_unary_s(embeddings, target, attributeSet))
	m_score = np.mean(targets_eval)
	return m_score, targets_eval

def _unary_s(embeddings, target, attributes):
	return np.mean([ spatial.distance.cosine(embeddings[target], embeddings[ai]) for ai in attributes ])







	