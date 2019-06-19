import string

import numpy as np
from gensim.models.keyedvectors import Word2VecKeyedVectors

from biasOps import project_onto_subspace

def load_legacy_w2v(w2v_file, dim=50):
    vectors = {}
    with open(w2v_file, 'r') as f:
        for line in f:
            vect = line.strip().rsplit()
            word = vect[0]
            vect = np.array([float(x) for x in vect[1:]])
            if(dim == len(vect)):
                vectors[word] = vect
        
    return vectors, dim

def load_legacy_w2v_as_keyvecs(w2v_file, dim=50):
    vectors = None
    with open(w2v_file, 'r') as f:
        vectors = Word2VecKeyedVectors(dim)

        ws = []
        vs = []

        for line in f:
            vect = line.strip().rsplit()
            word = vect[0]
            vect = np.array([float(x) for x in vect[1:]])
            if(dim == len(vect)):
                ws.append(word)
                vs.append(vect)
        vectors.add(ws, vs, replace=True)
    return vectors

def convert_legacy_to_keyvec(legacy_w2v):
    dim = len(legacy_w2v[legacy_w2v.keys()[0]])
    vectors = Word2VecKeyedVectors(dim)

    ws = []
    vs = []

    for word, vect in legacy_w2v.items():
        ws.append(word)
        vs.append(vect)
        assert(len(vect) == dim)
    vectors.add(ws, vs, replace=True)
    return vectors

def load_w2v(w2v_file, binary=True, limit=None):
    """
    Load Word2Vec format files using gensim and convert it to a dictionary
    """
    wv_from_bin = KeyedVectors.load_word2vec_format(w2v_file, binary=binary, limit=limit)
    dim = len(wv_from_bin[wv_from_bin.index2entity[0]])

    vectors = {w: wv_from_bin[w] for w in wv_from_bin.index2entity}

    return vectors, dim

def write_w2v(w2v_file, vectors):
    with open(w2v_file, 'w') as f:
        for word, vec in vectors.items():
            word = "".join(i for i in word if ord(i)<128)
            line = word + " " + " ".join([str(v) for v in vec]) + "\n"
            f.write(line)
        f.close()

def writeAnalogies(analogies, path):
    f = open(path, "w")
    f.write("Score,Analogy\n")
    for score, analogy, raw in analogies:
        f.write(str(score) + "," + str(analogy) + "," + str(raw) + "\n")
    f.close()

def writeGroupAnalogies(groups, path):
    f = open(path, "w")
    f.write("Score,Analogy\n")
    for analogies in groups:
        for score, analogy, raw in analogies:
            f.write(str(score) + "," + str(analogy) + "," + str(raw) + "\n")
    f.close()

def evalTerms(vocab, subspace, terms):
    for term in terms:
        vect = vocab[term]
        bias = project_onto_subspace(vect, subspace)
        print("Bias of '"+str(term)+"': {}".format(np.linalg.norm(bias)))

def pruneWordVecs(wordVecs):
    newWordVecs = {}
    for word, vec in wordVecs.items():
        valid=True
        if(not isValidWord(word)):
            valid = False
        if(valid):
            newWordVecs[word] = vec
    return newWordVecs

def preprocessWordVecs(wordVecs):
    """
    Following Bolukbasi:
    - only use the 50,000 most frequent words
    - only lower-case words and phrases
    - consisting of fewer than 20 lower-case characters
        (discard upper-case, digits, punctuation)
    - normalize all word vectors
    """
    newWordVecs = {}
    allowed = set(string.ascii_lowercase + ' ' + '_')

    for word, vec in wordVecs.items():
        chars = set(word)
        if chars.issubset(allowed) and len(word.replace('_', '')) < 20:
            newWordVecs[word] = vec / np.linalg.norm(vec)

    return newWordVecs

def removeWords(wordVecs, words):
    for word in words:
        if word in wordVecs:
            del wordVecs[word]
    return wordVecs

def isValidWord(word):
    return all([c.isalpha() for c in word])

def listContainsMultiple(source, target):
    for t in target:
        if(source[0] in t and source[1] in t):
            return True
    return False