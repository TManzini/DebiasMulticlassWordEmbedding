import json
from util import listContainsMultiple

def load_def_set_pairs(json_filepath):
	with open(json_filepath, "r") as f: 
		loadedData = json.load(f)
		data = {i: v for i, v in enumerate(loadedData["definite_sets"])}
		pairs = {}
		for i, (key, values) in enumerate(data.items()):
			pairs[i] = []
			#Load all the combos of pairs
			for v1 in values:
				for v2 in values:
					s = set([v1, v2])
					if(len(s) > 1 and not listContainsMultiple([v1, v2], pairs[i])):
						pairs[i].append([v1, v2])
			pairs[i] = [list(v) for v in pairs[i]]
		res = []
		for j in range(0, len(pairs[0])):
			res.append({})
			for k in pairs.keys():
				res[-1][k] = pairs[k][j]
		return res

def load_analogy_templates(json_filepath, mode):
	with open(json_filepath, "r") as f:	
		loadedData = json.load(f)
		return loadedData["analogy_templates"][mode]

def load_test_terms(json_filepath):
	with open(json_filepath, "r") as f:	
		loadedData = json.load(f)
		return loadedData["testTerms"]

def load_eval_terms(json_filepath, mode):
	with open(json_filepath, "r") as f:	
		loadedData = json.load(f)
		return loadedData["eval_targets"], loadedData["analogy_templates"][mode].values()

def load_def_sets(json_filepath):
	with open(json_filepath, "r") as f: 
		loadedData = json.load(f)
		return {i: v for i, v in enumerate(loadedData["definite_sets"])}