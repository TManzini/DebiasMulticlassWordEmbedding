# Debiasing Multiclass Word Embeddings

This repository contains the code that was written in support of the NAACL 2019 paper [Black is to Criminal as Caucasian is to Police:
Detecting and Removing Multiclass Bias in Word Embeddings](https://arxiv.org/abs/1904.04047).

The repository is broken into two components. 
1. Performing debiasing and MAC score calculations (./Debiasing)
2. Downstream evaluation (./Downstream)

## Debasing

In order to run these files several data files need to be downloaded. 
If you would like to replicate our results from scratch, you must download the following files.
1. The Reddit.l2 Corpus

If you would like to replicate our results we provide pretrained Word2Vec embeddings. 
1. Pretrained Baseline Word2Vecs
2. Word2Vecs which have been debiased using Equalize and Neutralize (Hard Debiasing) for gender, race, and religion. 
3. Word2Vecs which have been debiased using Soften for gender, race, and religion

If you are replicating our results from scratch, you can train your Word2Vecs using ./Debiasing/word2vec.py. Note that this file will generate Word2Vec embeddings for all the corpus files in the folder ./Debiasing/reddit.l2/*

Once you have trained your word embeddings you can evaluate you word embeddings using debais.py. Running debias.py requires the following command line arguments
* embeddingPath : The path to the word2vec embeddings
* vocabPath : The path to the social stereotype vocabulary
* mode : The type of social bias data that should be used when performing analogy generation and debiasing
* -hard : If this flag is set hard debiasing will be performed
* -soft : If this flag is used then soft debiasinng will be performed
* -w : If this flag is used then all the output of the analogy tasks, the debiased Word2Vecs and the MAC statistics will be written to disk in an folder named "./output"
* -v : If this flag is used then the debias script will execute in verbose mode
* -printLimit : An integer which defines how many of each type of analogy will be printed when executing in verbose mode

Example commands are included below for reference

This commmand performs hard religious debiasing based on attributes in the passed vocab file. Verbose mode is used and 100 analogies are printed for each embedding space (biased/debiased)
```
python debias.py 'data/w2vs/reddit.US.txt.tok.clean.cleanedforw2v.w2v' 'data/vocab/religion_attributes_optm.json' attribute -v -hard -printLimit 100
```
This commmand performs hard & soft gender debiasing based on roles in the passed vocab file. Verbose mode is used and 500 analogies are printed for each embedding space (biased/hard debiased/soft debiased)
```
python debias.py 'data/w2vs/reddit.US.txt.tok.clean.cleanedforw2v.w2v' 'data/vocab/gender_attributes_optm.json' role -v -hard -soft -printLimit 500
```

## Downstream
In order to replicate our results you must use the embeddings that were generated in the Debiasing section (or you can simply download our pretrained and predebiased embeddings). These embeddings should be stored in ./Downstream/data/wvs/. 

Additionally, you must download the (conll2003 dataset)[https://www.clips.uantwerpen.be/conll2003/ner/]. This data should be segmented into train, test, and val files which should be stored in ./Downstream/data/conll2003/. 

After these files have been placed in the appropriate locations, you can replicate our results by running the ipython notebook ./Downstream/BiasEvalPipelineRunner.ipynb

## Requirements
The following python packages are required.
* numpy
* scipy
* gensim
* sklearn
* pytorch
* jupyter
