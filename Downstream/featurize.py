import torch
import numpy as np

def formatBatchedExamples(batchedExamples, inputVocab, inputUnkToken, outputVocab):
	batchX = []
	batchY = []
	for x, y in batchedExamples:
		batchX.append([inputVocab.getIdx(i, inputUnkToken) for i in x])
		batchY.append(outputVocab.getIdx(y, None))
	batchX = torch.tensor(batchX).long()
	batchY = torch.tensor(batchY).long()
	return batchX, batchY

def constructEmbeddingTensorFromVocabAndWvs(wvs, vocab, embeddingSize):
	wvTensor = [[0.0]*embeddingSize for _ in range(len(vocab))]
	for i in range(len(vocab)):
		wvTensor[i] = wvs[vocab.getWord(i)]
	return torch.tensor(wvTensor)

def getExampleSubset(X, Y, xTerms):
	newX = []
	newY = []
	for x, y, in zip(X, Y):
		for term in xTerms:
			if term in x:
				newX.append(x)
				newY.append(y)
				#print("Adding", " ".join(x), "because it contains the term", term)
	return newX, newY

