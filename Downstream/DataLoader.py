

def loadNERDatasetXY(path, sourceField=0, targetField=1, sortLength=True):
	f = open(path, "r")
	lines = f.readlines()[2:]
	f.close()

	dataX = [[]]
	dataY = [[]]
	for line in lines:
		cl = cleanLine(line).split(" ")
		if(len(cl) > 1):
			dataX[-1].append(cl[sourceField])
			dataY[-1].append(cl[targetField])
		else:
			dataX.append([])
			dataY.append([])

	if(sortLength):
		data = zip(dataX, dataY)
		data = sorted(data, key=lambda x:-len(x[0]))
		dataX, dataY = zip(*data)

	return dataX, dataY

def cleanLine(line):
	return line.replace("\n", "").lstrip(" ").rstrip(" ")