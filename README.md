# bioasq10b-public

## What is this repository for? ###

This code implements Macquarie University's experiments and
participation in BioASQ 10b.
* [BioASQ](http://www.bioasq.org)
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

## How do I get set up? ###

Apart from the code in this repository, you will need the following files:

* `training10b.json` - available from [BioASQ](http://www.bioasq.org/)
* `rouge_10b.csv` - you can create it by running the following overnight:
```
>>> from classificationneural import saveRouge
>>> saveRouge('training10b.json', 'rouge_10b.csv',
               snippets_only = True)
```

Read the file `Dockerfile` for an idea of how to install the dependencies and
set up the system.

## Reading

* If you use this code, please cite the following paper:

D. Mollá (2022). Query-Focused Extractive Summarisation for Biomedical and COVID-19 Question Answering. *CLEF2022 Working Notes*.

## Examples of runs using pre-learnt models

To get the pre-learnt models used in the BioASQ10b runs, please email
diego.molla-aliod@mq.edu.au. The following models are available:

* task8b_distilbert_model_32.pt - for neural classification with DistilBERT and all BioASQ10b training data

* task8b_distilbert_alldata_model_32.pt - for neural classification with DistilBERT and all BioASQ10b training data


### DistilBERT

Using the entire training data:

```
>>> from classificationneural import bioasq_run
>>> bioasq_run(test_data='BioASQ-task10bPhaseB-testset1.json', model_type='distilbert_alldata', output_filename='bioasq-out-distilbert.json')
```

Using the last 50% of the training data:

```
>>> from classificationneural import bioasq_run
>>> bioasq_run(test_data='BioASQ-task10bPhaseB-testset1.json', model_type='distilbert', output_filename='bioasq-out-distilbert.json')
```


## Examples of cross-validation runs and their results

Below are 10-fold cross-validation results using the last 50% of the BioASQ10b training data.

```
rm diego.out; for F in 1 2 3 4 5 6 7 8 9 10; do python classificationneural.py -t DistilBERT --truncate_training --dropout 0.6 --nb_epoch 1 --batch_size 32 --fold $F >> diego.out; done
```

Remove the flag "truncate_training" to rrain on the entire training data

| Method | Training data | Batch size | Dropout | Epochs | Mean SU4 F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| DistilBERT | all | 32 | 0.6 | 1 | 0.275 |
| DistilBERT | last 50% | 32 | 0.6 | 1 | 0.311 |


## Who do I talk to? ###

Diego Molla: [diego.molla-aliod@mq.edu.au](mailto:diego.molla-aliod@mq.edu.au)
