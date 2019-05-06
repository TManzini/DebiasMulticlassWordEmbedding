import numpy as np

from biasOps import project_onto_subspace

from gensim.models.keyedvectors import Word2VecKeyedVectors

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

def evalTerms(vocab, subspace, terms):
    for term in terms:
        vect = vocab[term]
        bias = project_onto_subspace(vect, subspace)
        print "Bias of '"+str(term)+"': {}".format(np.linalg.norm(bias))

def pruneWordVecs(wordVecs):
    newWordVecs = {}
    for word, vec in wordVecs.items():
        valid=True
        if(not all([c.isalpha() for c in word])):
            valid = False
        if(valid):
            newWordVecs[word] = vec
    return newWordVecs