import argparse
import numpy as np

from util import write_w2v, writeAnalogies, writeGroupAnalogies, convert_legacy_to_keyvec, load_legacy_w2v, pruneWordVecs
from biasOps import identify_bias_subspace, neutralize_and_equalize, equalize_and_soften
from evalBias import generateAnalogies, multiclass_evaluation
from loader import load_def_sets, load_analogy_templates, load_test_terms, load_eval_terms
from scipy.stats import ttest_rel, spearmanr


parser = argparse.ArgumentParser()
parser.add_argument('embeddingPath')
parser.add_argument('vocabPath')
parser.add_argument('mode')
parser.add_argument('-hard', action='store_true')
parser.add_argument('-soft', action='store_true')
parser.add_argument('-w', action='store_true')
parser.add_argument('-v', action='store_true')
parser.add_argument('-k', type=int, default=1)
parser.add_argument('-printLimit', type=int, default=500)
parser.add_argument('-analogies', action="store_true")
args = parser.parse_args()

outprefix = args.vocabPath.replace("/", "_").replace("\\", "_").replace(".", "_")

print("Loading vocabulary from {}".format(args.vocabPath))

analogyTemplates = load_analogy_templates(args.vocabPath, args.mode)
defSets = load_def_sets(args.vocabPath)
testTerms = load_test_terms(args.vocabPath)

neutral_words = []
for value in analogyTemplates.values():
    neutral_words.extend(value)

print("Loading embeddings from {}".format(args.embeddingPath))
word_vectors, embedding_dim = load_legacy_w2v(args.embeddingPath)

print("Pruning Word Vectors... Starting with", len(word_vectors))
word_vectors = pruneWordVecs(word_vectors)
print("\tEnded with", len(word_vectors))

print("Identifying bias subspace")
subspace = identify_bias_subspace(word_vectors, defSets, args.k, embedding_dim)[:args.k]

if(args.hard):
    print("Neutralizing and Equalizing")
    new_hard_word_vectors = neutralize_and_equalize(word_vectors, neutral_words,
                        defSets.values(), subspace, embedding_dim)
if(args.soft):
    print("Equalizing and Softening")
    new_soft_word_vectors = equalize_and_soften(word_vectors, neutral_words,
                        defSets.values(), subspace, embedding_dim, verbose=args.v)

if(args.analogies):
    print("Generating Analogies")
    biasedAnalogies, biasedAnalogyGroups = generateAnalogies(analogyTemplates, convert_legacy_to_keyvec(word_vectors))
    if(args.hard):
        hardDebiasedAnalogies, hardDebiasedAnalogyGroups = generateAnalogies(analogyTemplates, convert_legacy_to_keyvec(new_hard_word_vectors))
    if(args.soft):
        softDebiasedAnalogies, softDebiasedAnalogyGroups = generateAnalogies(analogyTemplates, convert_legacy_to_keyvec(new_soft_word_vectors))

    if(args.w):
        print("Writing biased analogies to disk")
        writeAnalogies(biasedAnalogies, "output/" + outprefix + "_biasedAnalogiesOut.csv")
        writeGroupAnalogies(biasedAnalogyGroups, "output/" + outprefix + "_biasedAnalogiesOut_grouped.csv")

    if(args.v):
        print("Biased Analogies (0-" + str(args.printLimit) + ")")
        for score, analogy, _ in biasedAnalogies[:args.printLimit]:
            print(score, analogy)

    if(args.w):
        if(args.hard):
            print("Writing hard debiased analogies to disk")
            writeAnalogies(hardDebiasedAnalogies, "output/" + outprefix + "_hardDebiasedAnalogiesOut.csv")
            writeGroupAnalogies(hardDebiasedAnalogyGroups, "output/" + outprefix + "_hardDebiasedAnalogiesOut_grouped.csv")
        if(args.soft):
            print("Writing soft debiased analogies to disk")
            writeAnalogies(softDebiasedAnalogies, "output/" + outprefix + "_softDebiasedAnalogiesOut.csv")
            writeGroupAnalogies(softDebiasedAnalogyGroups, "output/" + outprefix + "_softDebiasedAnalogiesOut_grouped.csv")
    if(args.v):
        if(args.hard):
            print("="*20, "\n\n")
            print("Hard Debiased Analogies (0-" + str(args.printLimit) + ")")
            for score, analogy, _ in hardDebiasedAnalogies[:args.printLimit]:
                print(score, analogy)
        if(args.soft):
            print("="*20, "\n\n")
            print("Soft Debiased Analogies (0-" + str(args.printLimit) + ")")
            for score, analogy, _ in softDebiasedAnalogies[:args.printLimit]:
                print(score, analogy)
        
if(args.w):
    print("Writing data to disk")
    write_w2v("output/" + outprefix + "_" + args.mode + "_biasedEmbeddingsOut.w2v", word_vectors)
    if(args.hard):
        write_w2v("output/" + outprefix + "_" + args.mode + "_hardDebiasedEmbeddingsOut.w2v", new_hard_word_vectors)
    if(args.soft):
        write_w2v("output/" + outprefix + "_" + args.mode + "_softDebiasedEmbeddingsOut.w2v", new_soft_word_vectors)

print("Performing Evaluation")
evalTargets, evalAttrs = load_eval_terms(args.vocabPath, args.mode)

print("Biased Evaluation Results")
biasedMAC, biasedDistribution = multiclass_evaluation(word_vectors, evalTargets, evalAttrs)
print("Biased MAC:", biasedMAC)

if(args.hard):
    print("HARD Debiased Evaluation Results")
    debiasedMAC, debiasedDistribution = multiclass_evaluation(new_hard_word_vectors, evalTargets, evalAttrs)
    print("HARD MAC:", debiasedMAC)

    statistics, pvalue = ttest_rel(biasedDistribution, debiasedDistribution)
    print("HARD Debiased Cosine difference t-test", pvalue)

if(args.soft):
    print("SOFT Debiased Evaluation Results")
    debiasedMAC, debiasedDistribution = multiclass_evaluation(new_soft_word_vectors, evalTargets, evalAttrs)
    print("SOFT MAC:", debiasedMAC)

    statistics, pvalue = ttest_rel(biasedDistribution, debiasedDistribution)
    print("SOFT Debiased Cosine difference t-test", pvalue)

if(args.w):
    print("Writing statistics to disk")
    f = open("output/" + outprefix + "_statistics.csv", "w")
    f.write("Biased MAC,Debiased MAC,P-Value\n")
    f.write(str(biasedMAC) + "," +  str(debiasedMAC) + "," + str(pvalue) + "\n")
    f.close()