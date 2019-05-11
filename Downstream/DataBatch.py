import numpy as np
import torch

class SeqDataBatch:
	def __init__(self, X, Y, XVocab, YVocab):
		self.__X = list(X)
		self.__Y = list(Y)
		self.__XVocab = XVocab
		self.__YVocab = YVocab

		assert(len(X) == len(Y))
		self.__batchSize = len(X)

	def getNumericXY(self, getAs="torch", unkify=True):
		def __indexify(target, vocab):
			return vocab.getIdx(target, unkify=unkify)

		idxFunc = np.vectorize(__indexify)

		X = idxFunc(self.__X, self.__XVocab)
		Y = idxFunc(self.__Y, self.__YVocab)
		
		if(getAs.lower() == "torch" or getAs.lower() == "pytorch"):
			return torch.tensor(X), torch.tensor(Y)
		if(getAs.lower() == "numpy" or getAs.lower() == "np"):
			return np.array(X), np.array(Y)		
		return X, Y

	def padBatch(self, padValue, padX=True, padY=True, padingType="left"):
		padLeft = True if padingType.lower() == "left" else False
		if(padX):
			newX = []
			maxLenX = max([len(x) for x in self.__X])
			for x in self.__X:
				lenDiff = maxLenX - len(x)
				newX.append([padValue]*lenDiff*(padLeft) + x + [padValue]*lenDiff*(not padLeft))
			self.__X = newX

		if(padY):
			newY = []
			maxLenY = max([len(y) for y in self.__Y])
			for y in self.__Y:
				lenDiff = maxLenY - len(y)
				newY.append([padValue]*lenDiff*(padLeft) + y + [padValue]*lenDiff*(not padLeft))
			self.__Y = newY

	def addXY(self, newX, newY):
		self.__X.append(newX)
		self.__Y.append(newY)
		assert(len(self.__X) == len(self.__Y))
		self.__batchSize += 1
		
	def getX(self):
		return self.__X
	def getY(self):
		return self.__Y

	def getXVocab(self):
		return self.__XVocab
	def getYVocab(self):
		return self.__YVocab

	def setXVocab(self, newVocab):
		self.__XVocab = newVocab
		return True
	def setYVocab(self, newVocab):
		self.__YVocab = newVocab
		return True

	def getBatchSize(self):
		return self.__batchSize
	def __len__(self):
		return self.__batchSize