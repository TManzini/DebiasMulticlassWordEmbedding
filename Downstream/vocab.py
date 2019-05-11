class Vocab:
	def __init__(self, unkToken):
		self.__w2i = {}
		self.__i2w = []
		self.__curVocab = 0

		self.add(unkToken)
		self.__unkToken = unkToken

	def __len__(self):
		return self.__curVocab

	def add(self, word):
		res = 0
		try:
			return self.__w2i[word]
		except KeyError as e:
			self.__w2i[word] = self.__curVocab
			self.__i2w.append(word)
			res = self.__curVocab
			self.__curVocab += 1
		return res

	def getWord(self, i):
		return self.__i2w[i]

	def getIdx(self, word, unkify=True):
		if(not unkify):
			return self.__w2i[word]
		else:
			try:
				return self.__w2i[word]
			except KeyError as e:
				return self.__w2i[self.__unkToken]
