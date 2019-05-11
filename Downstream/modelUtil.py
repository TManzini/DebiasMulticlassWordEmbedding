import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score


def train_step(model, device, batchedData, optimizer, criterion, epoch, debugInterval=100):
    model.train()
    totalLoss = []
    totalExampleCount = 0
    for batch_idx, (data, target) in enumerate(batchedData):
        
        data = data.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        
        y_preds = model(data)      
        
        loss = criterion(y_preds, target)
        loss.backward()
        
        optimizer.step()
        
        if (batch_idx+1) % debugInterval == 0:
            print('Train Epoch: {} [{}/{}] \tLoss: {:.6f}'.format(epoch, batch_idx+1, len(batchedData), loss.item()))
        totalLoss.append(loss.item()*len(data))
        totalExampleCount += len(data)
    return sum(totalLoss)/totalExampleCount

def test(model, device, batchedData, criterion):
    model.eval()
    totalLoss = []
    totalExampleCount = 0
    for batch_idx, (data, target) in enumerate(batchedData):
        
        data = data.to(device)
        target = target.to(device)

        batchSize = len(data)
        
        y_preds = model(data)      
        
        loss = criterion(y_preds, target)
        
        #Store the losses
        totalLoss.append(loss.item()*batchSize)
        totalExampleCount += batchSize
    return sum(totalLoss)/totalExampleCount

def precisionRecallEval(model, device, batchedData):
    preds = []
    targets = []

    model.eval()
    for batch_idx, (data, target) in enumerate(batchedData):
        
        data = data.to(device)
        target = target.detach().numpy()
        
        y_preds = model.prod_forward(data).detach().numpy()
        assert(len(target)==len(y_preds))


        batchSize, seqLen, tagSpaceSize = y_preds.shape
        b = np.zeros((batchSize, seqLen, tagSpaceSize))
        for i, example in enumerate(target):
            for j, term in enumerate(example):
                #print(b.shape, i, j, term)
                b[i, j, term] = 1

        preds.append(y_preds)
        targets.append(b)

    preds = np.concatenate(preds, axis = None)
    targets = np.concatenate(targets, axis = None)

    precision = precision_score(targets, preds.round())
    recall = recall_score(targets, preds.round())
    f1 = f1_score(targets, preds.round(), average="macro")

    return precision, recall, f1, preds