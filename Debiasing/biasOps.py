import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

import torch

def normalize(word_vectors):
    for k, v in word_vectors.items():
        word_vectors[k] = v / np.linalg.norm(v)

def identify_bias_subspace(vocab, def_sets, subspace_dim, embedding_dim):
    """
    Similar to bolukbasi's implementation at
    https://github.com/tolga-b/debiaswe/blob/master/debiaswe/debias.py

    vocab - dictionary mapping words to embeddings
    def_sets - sets of words that represent extremes? of the subspace
            we're interested in (e.g. man-woman, boy-girl, etc. for binary gender)
    subspace_dim - number of vectors defining the subspace
    embedding_dim - dimensions of the word embeddings
    """
    # calculate means of defining sets
    means = {}
    for k, v in def_sets.items():
        wSet = []
        for w in v:
            try:
                wSet.append(vocab[w])
            except KeyError as e:
                pass
        set_vectors = np.array(wSet)
        means[k] = np.mean(set_vectors, axis=0)

    # calculate vectors to perform PCA
    matrix = []
    for k, v in def_sets.items():
        wSet = []
        for w in v:
            try:
                wSet.append(vocab[w])
            except KeyError as e:
                pass
        set_vectors = np.array(wSet)
        diffs = set_vectors - means[k]
        matrix.append(diffs)

    matrix = np.concatenate(matrix)

    pca = PCA(n_components=subspace_dim)
    pca.fit(matrix)

    return pca.components_

def project_onto_subspace(vector, subspace):
    v_b = np.zeros_like(vector)
    for component in subspace:
        v_b += np.dot(vector.transpose(), component) * component
    return v_b

def calculateDirectBias(vocab, neutral_words, bias_subspace, c=1):
    directBiasMeasure = 0
    for word in neutral_words:
        vec = vocab[word]
        directBiasMeasure += np.linalg.norm(cosine_similarity(vec, bias_subspace))**c
    directBiasMeasure *= 1.0/len(neutral_words)
    return directBiasMeasure

def neutralize_and_equalize(vocab, words, eq_sets, bias_subspace, embedding_dim):
    """
    vocab - dictionary mapping words to embeddings
    words - words to neutralize
    eq_sets - set of equality sets
    bias_subspace - subspace of bias from identify_bias_subspace
    embedding_dim - dimensions of the word embeddings
    """

    if bias_subspace.ndim == 1:
        bias_subspace = np.expand_dims(bias_subspace, 0)
    elif bias_subspace.ndim != 2:
        raise ValueError("bias subspace should be either a matrix or vector")

    new_vocab = vocab.copy()
    for w in words:
        # get projection onto bias subspace
        if w in vocab:
            v = vocab[w]
            v_b = project_onto_subspace(v, bias_subspace)

            new_v = (v - v_b) / np.linalg.norm(v - v_b)
            #print np.linalg.norm(new_v)
            # update embedding
            new_vocab[w] = new_v

    normalize(new_vocab)

    for eq_set in eq_sets:
        mean = np.zeros((embedding_dim,))

        #Make sure the elements in the eq sets are valid
        cleanEqSet = []
        for w in eq_set:
            try:
                _ = new_vocab[w]
                cleanEqSet.append(w)
            except KeyError as e:
                pass

        for w in cleanEqSet:
            mean += new_vocab[w]
        mean /= float(len(cleanEqSet))

        mean_b = project_onto_subspace(mean, bias_subspace)
        upsilon = mean - mean_b

        for w in cleanEqSet:
            v = new_vocab[w]
            v_b = project_onto_subspace(v, bias_subspace)

            frac = (v_b - mean_b) / np.linalg.norm(v_b - mean_b)
            new_v = upsilon + np.sqrt(1 - np.sum(np.square(upsilon))) * frac

            new_vocab[w] = new_v

    return new_vocab

def equalize_and_soften(vocab, words, eq_sets, bias_subspace, embedding_dim, l=0.2, verbose=True):
    vocabIndex, vocabVectors = zip(*vocab.items())
    vocabIndex = {i:label for i, label in enumerate(vocabIndex)}

    Neutrals = torch.tensor([vocab[w] for w in words]).float().t()

    Words = torch.tensor(vocabVectors).float().t()

    # perform SVD on W to reduce memory and computational costs
    # based on suggestions in supplementary material of Bolukbasi et al.
    u, s, _ = torch.svd(Words)
    s = torch.diag(s)

    # precompute
    t1 = s.mm(u.t())
    t2 = u.mm(s)

    Transform = torch.randn(embedding_dim, embedding_dim).float()
    BiasSpace = torch.tensor(bias_subspace).view(embedding_dim, -1).float()

    Neutrals.requires_grad = False
    Words.requires_grad = False
    BiasSpace.requires_grad = False
    Transform.requires_grad = True

    epochs = 10
    optimizer = torch.optim.SGD([Transform], lr=0.000001, momentum=0.0)

    for i in range(0, epochs):
        TtT = torch.mm(Transform.t(), Transform)
        norm1 = (t1.mm(TtT - torch.eye(embedding_dim)).mm(t2)).norm(p=2)

        norm2 = (Neutrals.t().mm(TtT).mm(BiasSpace)).norm(p=2)

        loss = norm1 + l * norm2
        norm1 = None
        norm2 = None

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if(verbose):
            print("Loss @ Epoch #" + str(i) + ":", loss)

    if(verbose):
        print("Optimization Completed, normalizing vector transform")

    debiasedVectors = {}
    for i, w in enumerate(Words.t()):
        transformedVec = torch.mm(Transform, w.view(-1, 1))
        debiasedVectors[vocabIndex[i]] = ( transformedVec / transformedVec.norm(p=2) ).detach().numpy().flatten()

    return debiasedVectors

def equalize_and_soften_old(vocab, words, eq_sets, bias_subspace, embedding_dim, l=0.2, verbose=True):
    vocabIndex, vocabVectors = zip(*vocab.items())
    vocabIndex = {i:label for i, label in enumerate(vocabIndex)}

    Neutrals = torch.tensor([vocab[w] for w in words]).float().t()

    Words = torch.tensor(vocabVectors).float().t()
    Transform = torch.randn(embedding_dim, embedding_dim).float()
    BiasSpace = torch.tensor(bias_subspace).view(embedding_dim, -1).float()

    Neutrals.requires_grad = False
    Words.requires_grad = False
    BiasSpace.requires_grad = False
    Transform.requires_grad = True

    epochs = 10
    optimizer = torch.optim.SGD([Transform], lr=0.000001, momentum=0.0)

    for i in range(0, epochs):
        TW = torch.mm(Transform, Words)
        WtW = torch.mm(Words.t(), Words)

        norm1 = (torch.mm(TW.t(), TW) - WtW).norm(p=2)
        TW = None
        WtW = None

        TNt = torch.mm(Transform, Neutrals).t()
        TB = torch.mm(Transform, BiasSpace)

        norm2 = torch.mm(TNt, TB).norm(p=2)
        TNt = None
        TB = None

        loss = norm1 + l * norm2
        norm1 = None
        norm2 = None

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if(verbose):
            print("Loss @ Epoch #" + str(i) + ":", loss)

    if(verbose):
        print("Optimization Completed, normalizing vector transform")

    debiasedVectors = {}
    for i, w in enumerate(Words.t()):
        transformedVec = torch.mm(Transform, w.view(-1, 1))
        debiasedVectors[vocabIndex[i]] = ( transformedVec / transformedVec.norm(p=2) ).detach().numpy().flatten()

    return debiasedVectors



