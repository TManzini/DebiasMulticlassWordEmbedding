import numpy as np
import math
import random

import torch.optim as optim
import torch.nn as nn
import torch

from vocab import Vocab
from featurize import formatBatchedExamples, constructEmbeddingTensorFromVocabAndWvs, getExampleSubset
from models.POSTagger import POSTagger
from modelUtil import train_step, test, precisionRecallEval

from DataLoader import loadNERDatasetXY
from DataBatch import SeqDataBatch
from scipy.stats import ttest_rel


#Data parameters (The whole pipeline will need to rerun if these are changed)
BATCH_SIZE = 64

MAX_SEQ_LEN = 128

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

#Learning Parameters (Only the model training code will need to rerun if these are changed)
LEARNING_RATE = 0.001
EPOCHS = 25
USE_CUDA = False
DEBUG_INTERVAL = 250
L2_REG = 0.001
MOMENTUM = 0.25

DEBIAS_EPS = 1e-10

def biasEvalPipeline(biasWvsPath, debiasWvsPath, nerPath, dataTarget, logFile=None, verbose=False, results=True):
    if(verbose):
        print("Loading CONLL2003 NER Data")
    trainX, trainY = loadNERDatasetXY(nerPath + "/train.txt", sourceField=0, targetField=dataTarget, sortLength=True)
    testX,  testY  = loadNERDatasetXY(nerPath + "/test.txt", sourceField=0, targetField=dataTarget, sortLength=True)
    valX,   valY   = loadNERDatasetXY(nerPath + "/valid.txt", sourceField=0, targetField=dataTarget, sortLength=True)

    if(verbose):
        print("Loading biased word vectors")
    biasedWvs = {}
    f = open(biasWvsPath)
    lines = f.readlines()
    f.close()
    for line in lines[1:]:
        wv = line.split(" ")
        biasedWvs[wv[0].lower()] = [float(v) for v in wv[1:]]
        
    if(verbose):
        print("Loading debiased word vectors")
    debiasedWvs = {}
    f = open(debiasWvsPath)
    lines = f.readlines()
    f.close()
    for line in lines[1:]:
        wv = line.replace("]","").replace("[","").split(" ")
        debiasedWvs[wv[0].lower()] = [float(v) for v in wv[1:]]
        EMBEDDING_SIZE = len([float(v) for v in wv[1:]])
        
    if(verbose):
        print("Handling wv Edge Cases")
    unk_wv = [float(v) for v in np.random.rand(EMBEDDING_SIZE)]
    pad_wv = [float(v) for v in np.random.rand(EMBEDDING_SIZE)]

    biasedWvs[UNK_TOKEN.lower()] = unk_wv
    biasedWvs[PAD_TOKEN.lower()] = pad_wv
    debiasedWvs[UNK_TOKEN.lower()] = unk_wv
    debiasedWvs[PAD_TOKEN.lower()] = pad_wv


    if(verbose):
        print("Detecting words that had their baises altered")
    debiasedTerms = []
    for key in biasedWvs.keys():
        totalDiff = sum([math.fabs(a-b) for a, b in zip(biasedWvs[key], debiasedWvs[key])])
        if(totalDiff > DEBIAS_EPS):
            debiasedTerms.append(key)

    if(verbose):
        print("Constructing test set containing examples with debiased terms.")
    testX_bias, testY_bias = getExampleSubset(testX, testY, debiasedTerms)


    if(verbose):
        print("Constructing input Vocabulary")
    inputVocab = Vocab(UNK_TOKEN.lower())
    for word in biasedWvs.keys():
        inputVocab.add(word.lower())
    inputVocab.add(PAD_TOKEN.lower())
    if(verbose):
        print("Constructed an input vocabulary with", len(inputVocab), "entries")

    if(verbose):
        print("Constructing output vocabulary")
    outputVocab = Vocab(UNK_TOKEN.lower())
    for ty in trainY:
        for label in ty:
            outputVocab.add(label.lower())
    outputVocab.add(PAD_TOKEN.lower())
    if(verbose):
        print("Constructed an output vocabulary with", len(outputVocab), "entries")


    if(verbose):
        print("Batching Training Data")

    batchedTrainData = []
    for i, (x, y) in enumerate(zip(trainX, trainY)):
        if(i % BATCH_SIZE == 0):
            batchedTrainData.append(SeqDataBatch([], [], inputVocab, outputVocab))
        batchedTrainData[-1].addXY(x, y)

    formattedBatchedTrainData = []
    for b in batchedTrainData:
        b.padBatch(PAD_TOKEN, padX=True, padY=True, padingType="left")
        x, y = b.getNumericXY("torch", unkify=True)
        formattedBatchedTrainData.append((x.long(), y.long()))
    if(verbose):
        print("Generated", len(formattedBatchedTrainData), "Train Batches")

    batchedValData = []
    for i, (x, y) in enumerate(zip(valX, valY)):
        if(i % BATCH_SIZE == 0):
            batchedValData.append(SeqDataBatch([], [], inputVocab, outputVocab))
        batchedValData[-1].addXY(x, y)

    formattedBatchedValData = []
    for b in batchedValData:
        b.padBatch(PAD_TOKEN, padX=True, padY=True, padingType="left")
        x, y = b.getNumericXY("torch", unkify=True)
        formattedBatchedValData.append((x.long(), y.long()))
    if(verbose):
        print("Generated", len(formattedBatchedValData), "Val Batches")

    batchedTestData = []
    for i, (x, y) in enumerate(zip(testX, testY)):
        if(i % BATCH_SIZE == 0):
            batchedTestData.append(SeqDataBatch([], [], inputVocab, outputVocab))
        batchedTestData[-1].addXY(x, y)

    formattedBatchedTestData = []
    for b in batchedTestData:
        b.padBatch(PAD_TOKEN, padX=True, padY=True, padingType="left")
        x, y = b.getNumericXY("torch", unkify=True)
        formattedBatchedTestData.append((x.long(), y.long()))
    if(verbose):
        print("Generated", len(formattedBatchedTestData), "Test Batches")

    batchedBiasTestData = []
    for i, (x, y) in enumerate(zip(testX_bias, testY_bias)):
        if(i % BATCH_SIZE == 0):
            batchedBiasTestData.append(SeqDataBatch([], [], inputVocab, outputVocab))
        batchedBiasTestData[-1].addXY(x, y)

    formattedBatchedBiasTestData = []
    for b in batchedBiasTestData:
        b.padBatch(PAD_TOKEN, padX=True, padY=True, padingType="left")
        x, y = b.getNumericXY("torch", unkify=True)
        formattedBatchedBiasTestData.append((x.long(), y.long()))
    if(verbose):
        print("Generated", len(formattedBatchedBiasTestData), "Test Bias Batches")


    # # Training from biased word vector

    posModel = POSTagger(EMBEDDING_SIZE, 20, len(inputVocab), len(outputVocab))
    posModel.setEmbeddings(constructEmbeddingTensorFromVocabAndWvs(biasedWvs, inputVocab, EMBEDDING_SIZE), freeze=True)

    device = torch.device("cuda" if USE_CUDA else "cpu")
    paramsToOptimize = filter(lambda p: p.requires_grad,posModel.parameters())
    optimizer = optim.RMSprop(paramsToOptimize, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=L2_REG)
    criterion = nn.CrossEntropyLoss()
    if(verbose):
        print("Starting Training")

    start = True
    bestValLoss = test(posModel, device, formattedBatchedValData, criterion)    
    for epoch in range(1, EPOCHS + 1):
        loss = train_step(posModel, device, formattedBatchedTrainData, optimizer, criterion, epoch, DEBUG_INTERVAL)
        val_loss = test(posModel, device, formattedBatchedValData, criterion)
        precision, recall, f1, _ = precisionRecallEval(posModel, device, formattedBatchedValData)
        if(verbose):
            print("Epoch #{} - \n\tVal Loss: {:.6f} \n\tVal Precision {:6f} \n\tVal Recall {:6f} \n\tVal Macro F1 {:6f}".format(epoch, val_loss, precision, recall, f1))
        if(val_loss < bestValLoss and not start):
            torch.save(posModel.state_dict(), "models/savedModels/model.m")
            bestValLoss = val_loss
        start = False

    if(verbose):
        print("Found best case validation loss to be " + str(bestValLoss) + "\n\t... Loading saved model and testing")
    posModel.load_state_dict(torch.load("models/savedModels/model.m"))
    test_loss = test(posModel, device, formattedBatchedTestData, criterion)
    precision, recall, f1, _ = precisionRecallEval(posModel, device, formattedBatchedTestData)
    if(verbose):
        print("TEST DATA -\n\tTest Loss: {:.6f} \n\tTest Precision {:6f} \n\tTest Recall {:6f} \n\tTest Macro F1 {:6f}".format(test_loss, precision, recall, f1))

    posModel.load_state_dict(torch.load("models/savedModels/model.m"))
    posModel.setEmbeddings(constructEmbeddingTensorFromVocabAndWvs(biasedWvs, inputVocab, EMBEDDING_SIZE), freeze=True)
    biased_precision, biased_recall, biased_f1, biasedPreds = precisionRecallEval(posModel, device, formattedBatchedBiasTestData)
    test_biased_loss = test(posModel, device, formattedBatchedBiasTestData, criterion)
    if(logFile != None):
        logFile.write("Embedding Swap results for:\n\t" + biasWvsPath + "\n\t" + debiasWvsPath + "\n")
        logFile.flush()
    if(results):
        print("============= BIASED EMBEDDINGS TEST RESULTS =============")
        print("loss: " + str(test_biased_loss))
        print("precision: " + str(biased_precision))
        print("recall: " + str(biased_recall))
        print("f1: " + str(biased_f1))
    if(logFile != None):
        logFile.write("============= BIASED EMBEDDINGS TEST RESULTS =============" + "\n")
        logFile.write("loss: " + str(test_biased_loss) + "\n")
        logFile.write("precision: " + str(biased_precision) + "\n")
        logFile.write("recall: " + str(biased_recall) + "\n")
        logFile.write("f1: " + str(biased_f1) + "\n")
        logFile.flush()

    posModel.load_state_dict(torch.load("models/savedModels/model.m"))
    posModel.setEmbeddings(constructEmbeddingTensorFromVocabAndWvs(debiasedWvs, inputVocab, EMBEDDING_SIZE), freeze=True)
    debiased_precision, debiased_recall, debiased_f1, debiasedPreds = precisionRecallEval(posModel, device, formattedBatchedBiasTestData)
    test_debiased_loss = test(posModel, device, formattedBatchedBiasTestData, criterion)

    statistics, pvalue = ttest_rel(biasedPreds, debiasedPreds)

    if(results):
        print("============= DEBIASED EMBEDDINGS TEST RESULTS =============")
        print("loss: " + str(test_debiased_loss))
        print("precision: " + str(debiased_precision))
        print("recall: " + str(debiased_recall))
        print("f1: " + str(debiased_f1))
    if(logFile != None):
        logFile.write("============= DEBIASED EMBEDDINGS TEST RESULTS =============" + "\n")
        logFile.write("loss: " + str(test_debiased_loss) + "\n")
        logFile.write("precision: " + str(debiased_precision) + "\n")
        logFile.write("recall: " + str(debiased_recall) + "\n")
        logFile.write("f1: " + str(debiased_f1) + "\n")
        logFile.flush()

    if(results):
        print("============= EMBEDDINGS COMPARISION RESULTS =============")
        print("delta loss: " + str(test_debiased_loss - test_biased_loss))
        print("delta precision: " + str(debiased_precision - biased_precision))
        print("delta recall: " + str(debiased_recall - biased_recall))
        print("delta f1: " + str(debiased_f1 - biased_f1))
        print("p-value: " + str(pvalue) + "\n")
    if(logFile != None):
        logFile.write("============= EMBEDDINGS COMPARISION RESULTS =============" + "\n")
        logFile.write("delta loss: " + str(test_debiased_loss - test_biased_loss) + "\n")
        logFile.write("delta precision: " + str(debiased_precision - biased_precision) + "\n")
        logFile.write("delta recall: " + str(debiased_recall - biased_recall) + "\n")
        logFile.write("delta f1: " + str(debiased_f1 - biased_f1) + "\n")
        logFile.write("p-value: " + str(pvalue) + "\n")
        logFile.flush()


    # # Training from debiased word vector


    debias_posModel = POSTagger(EMBEDDING_SIZE, 20, len(inputVocab), len(outputVocab))
    debias_posModel.setEmbeddings(constructEmbeddingTensorFromVocabAndWvs(debiasedWvs, inputVocab, EMBEDDING_SIZE), freeze=True)


    device = torch.device("cuda" if USE_CUDA else "cpu")
    paramsToOptimize = filter(lambda p: p.requires_grad,debias_posModel.parameters())
    optimizer = optim.RMSprop(paramsToOptimize, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=L2_REG)
    criterion = nn.CrossEntropyLoss()
    if(verbose):
        print("Starting Training")

    start = True
    bestValLoss = test(debias_posModel, device, formattedBatchedValData, criterion)   
    for epoch in range(1, EPOCHS + 1):
        loss = train_step(debias_posModel, device, formattedBatchedTrainData, optimizer, criterion, epoch, DEBUG_INTERVAL)
        val_loss = test(debias_posModel, device, formattedBatchedValData, criterion)
        precision, recall, f1, _ = precisionRecallEval(debias_posModel, device, formattedBatchedValData)
        if(verbose):
            print("Epoch #{} - \n\tVal Loss: {:.6f} \n\tVal Precision {:6f} \n\tVal Recall {:6f} \n\tVal Macro F1 {:6f}".format(epoch, val_loss, precision, recall, f1))
        if(val_loss < bestValLoss and not start):
            torch.save(debias_posModel.state_dict(), "models/savedModels/debiased_model.m")
            bestValLoss = val_loss
        start = False

    if(verbose):
        print("Found best case validation loss to be " + str(bestValLoss) + "\n... Loading saved model and testing")
    debias_posModel.load_state_dict(torch.load("models/savedModels/debiased_model.m"))
    test_loss = test(debias_posModel, device, formattedBatchedTestData, criterion)
    precision, recall, f1, _ = precisionRecallEval(debias_posModel, device, formattedBatchedTestData)
    if(verbose):
        print("TEST DATA -\n\tTest Loss: {:.6f} \n\tTest Precision {:6f} \n\tTest Recall {:6f} \n\tTest Macro F1 {:6f}".format(test_loss, precision, recall, f1))
      


    posModel.load_state_dict(torch.load("models/savedModels/model.m"))
    a_biased_precision, a_biased_recall, a_biased_f1, biasedPreds = precisionRecallEval(posModel, device, formattedBatchedBiasTestData)
    a_test_biased_loss = test(posModel, device, formattedBatchedBiasTestData, criterion)
    if(logFile != None):
        logFile.write("Retrain results for:\n\t" + biasWvsPath + "\n\t" + debiasWvsPath + "\n")
        logFile.flush()
    if(results):
        print("============= BIASED MODEL TEST RESULTS =============")
        print("loss: " + str(a_test_biased_loss))
        print("precision: " + str(a_biased_precision))
        print("recall: " + str(a_biased_recall))
        print("f1: " + str(a_biased_f1))
    if(logFile != None):
        logFile.write("============= BIASED MODEL TEST RESULTS =============" + "\n")
        logFile.write("loss: " + str(a_test_biased_loss) + "\n")
        logFile.write("precision: " + str(a_biased_precision) + "\n")
        logFile.write("recall: " + str(a_biased_recall) + "\n")
        logFile.write("f1: " + str(a_biased_f1) + "\n")
        logFile.flush()

    debias_posModel.load_state_dict(torch.load("models/savedModels/debiased_model.m"))
    a_debiased_precision, a_debiased_recall, a_debiased_f1, debiasedPreds = precisionRecallEval(debias_posModel, device, formattedBatchedBiasTestData)
    a_test_debiased_loss = test(debias_posModel, device, formattedBatchedBiasTestData, criterion)

    statistics, pvalue = ttest_rel(biasedPreds, debiasedPreds)

    if(results):
        print("============= DEBIASED MODEL TEST RESULTS =============")
        print("loss: " + str(a_test_debiased_loss))
        print("precision: " + str(a_debiased_precision))
        print("recall: " + str(a_debiased_recall))
        print("f1: " + str(a_debiased_f1))
    if(logFile != None):
        logFile.write("============= DEBIASED MODEL TEST RESULTS =============" + "\n")
        logFile.write("loss: " + str(a_test_debiased_loss) + "\n")
        logFile.write("precision: " + str(a_debiased_precision) + "\n")
        logFile.write("recall: " + str(a_debiased_recall) + "\n")
        logFile.write("f1: " + str(a_debiased_f1) + "\n")
        logFile.flush()

    if(results):
        print("============= MODEL COMPARISION RESULTS =============")
        print("delta loss: " + str(a_test_debiased_loss - a_test_biased_loss))
        print("delta precision: " + str(a_debiased_precision - a_biased_precision))
        print("delta recall: " + str(a_debiased_recall - a_biased_recall))
        print("delta f1: " + str(a_debiased_f1 - a_biased_f1))
        print("p-value: " + str(pvalue))

    if(logFile != None):
        logFile.write("============= MODEL COMPARISION RESULTS =============")
        logFile.write("delta loss: " + str(a_test_debiased_loss - a_test_biased_loss) + "\n")
        logFile.write("delta precision: " + str(a_debiased_precision - a_biased_precision) + "\n")
        logFile.write("delta recall: " + str(a_debiased_recall - a_biased_recall) + "\n")
        logFile.write("delta f1: " + str(a_debiased_f1 - a_biased_f1) + "\n")
        logFile.write("p-value: " + str(pvalue) + "\n")
        logFile.flush()
