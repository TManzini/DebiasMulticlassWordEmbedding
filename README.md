# Debiasing Multiclass Word Embeddings

This repository contains the code that was written in support of the NAACL 2019 paper [Black is to Criminal as Caucasian is to Police:
Detecting and Removing Multiclass Bias in Word Embeddings](https://arxiv.org/abs/1904.04047).

The repository has three main components. 
1. Performing debiasing and MAC score calculations (./Debiasing/debias.py)
2. Cluster bias analysis (./Debiasing/neighborAnalysis.py)
3. Downstream evaluation (./Downstream/BiasEvalPipelineRunner.ipynb)

## Debiasing & MAC

In order to run these files several data files need to be downloaded. 
If you would like to replicate our results from scratch, you must download the following files.
1. [The Reddit.l2 Corpus](http://cl.haifa.ac.il/projects/L2/)

If you would like to replicate our results without training your own Word2Vec embeddings, we provide pretrained Word2Vec embeddings. Note that the results in the paper were on w2v_0.
1. Pretrained Baseline Word2Vecs [w2v_0](https://drive.google.com/file/d/1IJdGfnKNaBLHP9hk0Ns7kReQwo_jR1xx/view?usp=sharing), [w2v_1](https://drive.google.com/file/d/1gDXFBFcOJuRTrTveBYnW5vH0uSSATwp_/view?usp=sharing), [w2v_2](https://drive.google.com/file/d/102grp_w69V91OuLIgY9aEXWbjEWAx3qD/view?usp=sharing), [w2v_3](https://drive.google.com/file/d/1IO6gucgEVxxzNPKrdARO6KDYbBBIwBjM/view?usp=sharing), [w2v_4](https://drive.google.com/file/d/1IhdRfHg373OYP_c-wsxEddxWIRpIlpNH/view?usp=sharing) 
2. Word2Vecs which have been debiased using hard debiasing for [gender](https://drive.google.com/open?id=1UwihHVAP7IykOQDPrT4BOFOlgKqJUU-l), [race](https://drive.google.com/open?id=1at-OZonjKtb-Z1MvvLX3embAbZyfAmwX), and [religion](https://drive.google.com/open?id=13g_3ci859OS-ZkfRuMd6qdfDXe4bTQXP) - All based on w2v_0. 
3. Word2Vecs which have been debiased using soft debiasing for [gender](https://drive.google.com/file/d/1JAGTYfH9I0pZ-UA8BdJq-AmswaXovRuM/view?usp=sharing), [race](https://drive.google.com/file/d/1-2JYInfa4vYqqniqMHTmeVCR0h3dgQ2T/view?usp=sharing), and [religion](https://drive.google.com/file/d/11g5u1S8TW6S7hELlM9-MIBrPbsqobqaq/view?usp=sharing) - All based on w2v_0.

If you are replicating our results from scratch, you can train your Word2Vecs using ./Debiasing/word2vec.py. Note that this file will generate Word2Vec embeddings for all the corpus files in the folder ./Debiasing/reddit.l2/*

Once you have trained your word embeddings you can evaluate you word embeddings using debais.py. Running debias.py requires the following command line arguments
* embeddingPath : The path to the word2vec embeddings
* vocabPath : The path to the social stereotype vocabulary
* mode : The type of social bias data that should be used when performing analogy generation and debiasing
* -hard : If this flag is set hard debiasing will be performed
* -soft : If this flag is used then soft debiasinng will be performed
* -k : An integer which denotes how many principal components that are used to define the bias subspace (Defaults to 1).
* -w : If this flag is used then all the output of the analogy tasks, the debiased Word2Vecs and the MAC statistics will be written to disk in an folder named "./output"
* -v : If this flag is used then the debias script will execute in verbose mode
* -analogies : If this flag is used then analogies will be generated.
* -printLimit : An integer which defines how many of each type of analogy will be printed when executing in verbose mode

Our vocabularies for bias detection and removal can be found under ./Debiasing/data/vocab.

Example commands are included below for reference

This commmand performs hard religious debiasing based on attributes in the passed vocab file. Verbose mode is used and the first 2 PCA components are used for debiasing
```
python debias.py 'data/w2vs/reddit.US.txt.tok.clean.cleanedforw2v.w2v' 'data/vocab/religion_attributes_optm.json' attribute -v -hard -k 2
```
This commmand performs hard & soft gender debiasing based on roles in the passed vocab file. Verbose mode is used and 500 analogies are printed for each embedding space (biased/hard debiased/soft debiased)
```
python debias.py 'data/w2vs/reddit.US.txt.tok.clean.cleanedforw2v.w2v' 'data/vocab/gender_attributes_optm.json' role -v -hard -soft -printLimit 500 -analogies
```

## Cluster Bias Analysis
To run the cluster bias analysis (based on [Gonen and Goldberg (2019)](https://arxiv.org/pdf/1903.03862.pdf)), run the following:
```
python neighborAnalysis.py <biased embeddings> <debiased embeddings> <debiasing info> <targets>
```

For example:
```
python neighborAnalysis.py 'data/w2vs/reddit.US.txt.tok.clean.cleanedforw2v.w2v' 'output/data_race_attributes_optm_json_role_hardDebiasedEmbeddingsOut.w2v' 'data/vocab/race_attributes_optm.json' 'professions.json' --multi
```
Arguments/flags:
- targets: a JSON file containing a list of target words to evaluate bias on, such as Bolukbasi's [`professions.json`](https://github.com/tolga-b/debiaswe/blob/master/data/professions.json)
- `--bias_specific`: a JSON file containing a list of words that inherently contain bias (for example, he and she for gender) and should be ignored. For example, [here](https://github.com/tolga-b/debiaswe/blob/master/data/gender_specific_full.json]), or `data/vocab/religion_specific.json`.
- `--multi`: Set for multiclass debiasing
- `-v`: Verbose mode

## Downstream Evaluation
In order to replicate our results you must use the embeddings that were generated in the Debiasing section (or you can simply download our pretrained and predebiased embeddings). These embeddings should be stored in ./Downstream/data/wvs/. 

Additionally, you must download the [conll2003 dataset](https://www.clips.uantwerpen.be/conll2003/ner/). This data should be segmented into train, test, and val files which should be stored in ./Downstream/data/conll2003/. 

After these files have been placed in the appropriate locations, you can replicate our results by running the ipython notebook ./Downstream/BiasEvalPipelineRunner.ipynb

## Requirements
The following python packages are required (Python 2).
* numpy 1.14.5
* scipy 1.1.0
* gensim 3.5.0
* sklearn 0.19.2
* pytorch 0.4.0
* matplotlib 2.2.3
* jupyter 1.0.0

## Citing
If you found this repository or our paper helpful please consider citing us with this bibtex.  
```
@article{manzini2019black,
  title={Black is to Criminal as Caucasian is to Police: Detecting and Removing Multiclass Bias in Word Embeddings},
  author={Manzini, Thomas and Lim, Yao Chong and Tsvetkov, Yulia and Black, Alan W},
  journal={arXiv preprint arXiv:1904.04047},
  year={2019}
}
```
