from scipy import spatial
import numpy as np
import itertools

def generateAnalogies_parallelogram(analogyTemplates, keyedVecs):
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

def scoredAnalogyAnswers(a,b,x, keyedVecs, thresh=12.5):
	words = [w for w in keyedVecs.vocab if np.linalg.norm(np.array(keyedVecs[w])-np.array(keyedVecs[x])) < thresh]

	def cos(a,b,x,y):
		aVec = np.array(keyedVecs[a])
		bVec = np.array(keyedVecs[b])
		xVec = np.array(keyedVecs[x])
		yVec = np.array(keyedVecs[y])
		numerator = (aVec-bVec).dot(xVec-yVec)
		denominator = np.linalg.norm(aVec-bVec)*np.linalg.norm(xVec-yVec)
		return numerator/(denominator if denominator != 0 else 1e-6)

	return sorted([(cos(a,b,x,y), a,b,x,y) for y in words], reverse=True)

def generateAnalogies(analogyTemplates, keyedVecs):
	expandedAnalogyTemplates = []
	for A, stereotypes in analogyTemplates.items():
		for B, _ in analogyTemplates.items():
			if(A != B):
				for stereotype in stereotypes:
					expandedAnalogyTemplates.append([A, stereotype, B])

	analogies = []
	outputGroups = []
	for a,b,x in expandedAnalogyTemplates:
		outputs = scoredAnalogyAnswers(a,b,x,keyedVecs)
		formattedOutput = []
		
		for score, a_w, b_w, x_w, y_w in outputs:
			
			analogy = str(a_w) + " is to " + str(b_w) + " as " + str(x_w) + " is to " + str(y_w)
			analogyRaw = [a_w, b_w, x_w, y_w]
			analogies.append([score, analogy, analogyRaw])
			formattedOutput.append([score, analogy, analogyRaw])
		outputGroups.append(formattedOutput)

	analogies = sorted(analogies, key=lambda x:-x[0])
	return analogies, outputGroups

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







	