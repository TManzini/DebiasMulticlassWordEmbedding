import torch
import torch.nn as nn
import torch.nn.functional as F

class POSTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, bidirectional=True):
        super(POSTagger, self).__init__()
        self.__hidden_dim = hidden_dim
        self.__directions = 2 if bidirectional else 1


        self.__encoder = nn.Embedding(vocab_size, embedding_dim)

        self.__rnn = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidirectional, batch_first=True)

        self.__pred = nn.Linear(hidden_dim*self.__directions, tagset_size)

    def forward(self, batch):
        batchSize, seqLen  = batch.size()
        embeds     = self.__encoder(batch)
        rnn_out, _ = self.__rnn(embeds)
        rnn_out    = rnn_out.contiguous().view(-1, self.__hidden_dim*self.__directions)
        tagSpace   = self.__pred(rnn_out)
        tagSpace   = tagSpace.contiguous().view(batchSize, seqLen, -1)
        y_preds    = F.log_softmax(tagSpace, dim=2)
        y_preds    = y_preds.permute(0,2,1)
        return y_preds

    def prod_forward(self, batch):
        batchSize, seqLen  = batch.size()
        embeds     = self.__encoder(batch)
        rnn_out, _ = self.__rnn(embeds)
        rnn_out    = rnn_out.contiguous().view(-1, self.__hidden_dim*self.__directions)
        tagSpace   = self.__pred(rnn_out)
        tagSpace   = tagSpace.contiguous().view(batchSize, seqLen, -1)
        y_preds    = F.softmax(tagSpace, dim=2)
        return y_preds

    def setEmbeddings(self, embeddings, freeze=True):
        self.__encoder.weight.data.copy_(embeddings)
        self.__encoder.weight.requires_grad=not freeze