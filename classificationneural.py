"""classification.py -- perform classification-based summarisation using deep learning architectures

Author: Diego Molla <dmollaaliod@gmail.com>
Created: 16/1/2019
"""

import json
import codecs
import csv
import sys
import os
import shutil
#import re
import random
import glob
from subprocess import Popen, PIPE
from multiprocessing import Pool
from functools import partial
import numpy as np
import progressbar

from sklearn.model_selection import KFold

from my_tokenizer import my_sent_tokenize as sent_tokenize

from xml_abstract_retriever import getAbstract
from nnmodels import nnc
from summariser.basic import answersummaries

def bioasq_train(small_data=False, verbose=2, model_type='distilbert'):
    """Train model for BioASQ"""
    
    assert model_type in ['bert', 'biobert', 'distilbert', 'albert', 'qaalbert']
    if model_type == 'bert':
        rouge_labels = False
        classification_type = 'BERT'
        nb_epoch = 8
        dropout = 0.8
        batch_size=32
        savepath = "./task10b_bert_model_32.pt"
    elif model_type == 'biobert':
        rouge_labels = False
        classification_type = 'BioBERT'
        nb_epoch = 1
        dropout = 0.7
        batch_size=32
        savepath = "./task10b_biobert_model_32.pt"
    elif model_type == 'distilbert':
        rouge_labels = False
        classification_type = 'DistilBERT'
        nb_epoch = 1
        dropout = 0.6
        batch_size=32
        savepath = "./task10b_distilbert_model_32.pt"
    elif model_type == 'albert':
        rouge_labels = False
        classification_type = 'ALBERT'
        nb_epoch = 5
        dropout = 0.5
        batch_size=32
        savepath = "./task10b_albert_model_32.pt"
    elif model_type == 'qaalbert':
        rouge_labels = False
        classification_type = 'QA_ALBERT'
        nb_epoch = 5
        dropout = 0.4
        batch_size=32
        savepath = "./task10b_qaalbert8b_model_32.pt"

    if small_data:
        nb_epoch = 3

    print("Training for BioASQ", model_type, "with batch size", batch_size)
    classifier = Classification('training10b.json', #_reranked.json',
                                'rouge_10b.csv',
                                #'train7b_filtered_v1.json',
                                #'rouge_train7b_dima.csv',
                                rouge_labels=rouge_labels,
                                nb_epoch=nb_epoch,
                                verbose=verbose,
                                classification_type=classification_type,
                                hidden_layer=50,
                                dropout=dropout,
                                batch_size=batch_size)

    #indices = list(range(len(classifier.data)))
    size_data = len(classifier.data)
    percent = 50
    #indices = list(range(0,size_data*percent/100)))
    indices = list(range(int((100-percent)*size_data/100), size_data))

    if small_data:
        print("Training bioasq with small data")
        indices = indices[:20]

    classifier.train(indices)
    classifier.save(savepath)

def bioasq_run(nanswers={"summary": 6,
                         "factoid": 2,
                         "yesno": 2,
                         "list": 3},
                test_data='8B1_golden.json',
               model_type='bert',
               output_filename='bioasq-out-nnc.json'):
    """Run model for BioASQ"""
    
    assert model_type in ['bert', 'biobert', 'distilbert', 'distilbert_alldata', 'albert', 'qaalbert']
    if model_type == 'bert':
        rouge_labels = False
        classification_type = 'BERT'
        nb_epoch = 8
        dropout = 0.8
        batch_size=32
        loadpath = "./task10b_bert_model_32.pt"
    elif model_type == 'biobert':
        rouge_labels = False
        classification_type = 'BioBERT'
        nb_epoch = 1
        dropout = 0.7
        batch_size=32
        loadpath = "./task10b_biobert_model_32.pt"
    elif model_type == 'distilbert':
        rouge_labels = False
        classification_type = 'DistilBERT'
        nb_epoch = 1
        dropout = 0.6
        batch_size=32
        loadpath = "./task10b_distilbert_model_32.pt"
    elif model_type == 'distilbert_alldata':
        rouge_labels = False
        classification_type = 'DistilBERT'
        nb_epoch = 1
        dropout = 0.6
        batch_size=32
        loadpath = "./task10b_distilbert_alldata_model_32.pt"
    elif model_type == 'albert':
        rouge_labels = False
        classification_type = 'ALBERT'
        nb_epoch = 5
        dropout = 0.5
        batch_size=32
        loadpath = "./task10b_albert_model_32.pt"
    elif model_type == 'qaalbert':
        rouge_labels = False
        classification_type = 'QA_ALBERT'
        nb_epoch = 5
        dropout = 0.4
        batch_size=32
        loadpath = "./task10b_qaalbert8b_model_32.pt"


    print("Running bioASQ")
    classifier = Classification('training10b.json', #_reranked.json',
                                'rouge_10b.csv',
                                nb_epoch=nb_epoch,
                                rouge_labels=rouge_labels,
                                verbose=2,
                                classification_type=classification_type,
                                hidden_layer=50,
                                dropout=dropout,
                                batch_size=batch_size)
#    indices = list(range(len(classifier.data)))
#    classifier.train(indices, savepath=savepath, restore_model=True)
    classifier.load(loadpath)
    testset = load_test_data(test_data)
    print("LOADED")
    answers = yield_bioasq_answers(classifier,
                                   testset,
                                   nanswers={"summary": 6,
                                             "factoid": 2,
                                             "yesno": 2,
                                             "list": 3})
    result = {"questions": [a for a in answers]}
    print("Saving results in file %s" % output_filename)
    with open(output_filename, 'w') as f:
        f.write(json.dumps(result, indent=2))

def loaddata(filename):
    """Load the JSON data
    >>> data = loaddata('training10b.json')
    Loading training10b.json
    >>> len(data)
    4234
    >>> sorted(data[0].keys())
    ['body', 'concepts', 'documents', 'id', 'ideal_answer', 'snippets', 'type']
    """
    print("Loading", filename)
    data = json.load(open(filename, encoding="utf-8"))
    return [x for x in data['questions'] if 'ideal_answer' in x]

def load_test_data(filename):
    """Load the JSON data
    >>> data = loaddata('training10b.json')
    Loading training10b.json
    >>> len(data)
    4234
    >>> sorted(data[0].keys())
    ['body', 'concepts', 'documents', 'id', 'ideal_answer', 'snippets', 'type']
    """
    print("Loading", filename)
    data = json.load(open(filename, encoding="utf-8"))
    return data['questions']

def yield_candidate_text(questiondata, snippets_only=True):
    """Yield all candidate text for a question
    >>> data = loaddata("training10b.json")
    Loading training10b.json
    >>> y = yield_candidate_text(data[0], snippets_only=True)
    >>> next(y)
    ('55031181e9bde69634000014', 0, 'Hirschsprung disease (HSCR) is a multifactorial, non-mendelian disorder in which rare high-penetrance coding sequence mutations in the receptor tyrosine kinase RET contribute to risk in combination with mutations at other genes')
    >>> next(y)
    ('55031181e9bde69634000014', 1, "In this study, we review the identification of genes and loci involved in the non-syndromic common form and syndromic Mendelian forms of Hirschsprung's disease.")
    """
    past_pubmed = set()
    sn_i = 0
    for sn in questiondata['snippets']:
        if snippets_only:
            for s in sent_tokenize(sn['text']):
                yield (questiondata['id'], sn_i, s)
                sn_i += 1
            continue

        pubmed_id = os.path.basename(sn['document'])
        if pubmed_id in past_pubmed:
            continue
        past_pubmed.add(pubmed_id)
        file_name = os.path.join("Task6bPubMed", pubmed_id+".xml")
        sent_i = 0
        for s in sent_tokenize(getAbstract(file_name, version="0")[0]):
            yield (pubmed_id, sent_i, s)
            sent_i += 1

def yield_bioasq_answers(classifier, testset, nanswers=3):
    """Yield answer of each record for BioASQ shared task"""
    with progressbar.ProgressBar(max_value=len(testset)) as bar:
        for i, r in enumerate(testset):
            test_question = r['body']
            test_id = r['id']
            test_candidates = [(sent, sentid)
                            for (pubmedid, sentid, sent)
                            in yield_candidate_text(r)]
    #        test_snippet_sentences = [s for snippet in r['snippets']
    #                                  for s in sent_tokenize(snippet['text'])]
            if len(test_candidates) == 0:
                print("Warning: no text to summarise")
                test_summary = ""
            else:
                if isinstance(nanswers,dict):
                    n = nanswers[r['type']]
                else:
                    n = nanswers
                test_summary = " ".join(classifier.answersummaries([(test_question,
                                                                    test_candidates,
                                                                    n)])[0])
                #print("Test summary:", test_summary)

            if r['type'] == "yesno":
                exactanswer = "yes"
            else:
                exactanswer = ""

            yield {"id": test_id,
                "ideal_answer": test_summary,
                "exact_answer": exactanswer}
            bar.update(i)

def collect_one_item(this_index, indices, testindices, data, labels):
    "Collect one item for parallel processing"
    qi, d = this_index
    if qi in indices:
        partition = 'main'
    elif testindices != None and qi in testindices:
        partition = 'test'
    else:
        return None

    this_question = d['body']

    if 'snippets' not in d:
        return None
    data_snippet_sentences = [s for sn in d['snippets']
                              for s in sent_tokenize(sn['text'])]

    if len(data_snippet_sentences) == 0:
        return None

    candidates_questions = []
    candidates_sentences = []
    candidates_sentences_ids = []
    label_data = []
    for pubmed_id, sent_id, sent in yield_candidate_text(d):
        candidates_questions.append(this_question)
        candidates_sentences.append(sent)
        candidates_sentences_ids.append(sent_id)
        label_data.append(labels[(qi, pubmed_id, sent_id)])

    return partition, label_data, candidates_questions, candidates_sentences, candidates_sentences_ids

def yieldRouge(corpus_data, xml_rouge_filename="rouge.xml",
               snippets_only=True):
    """yield ROUGE scores of all sentences in corpus
    >>> data = loaddata('training10b.json')
    Loading training10b.json
    >>> rouge = yieldRouge(data)
    >>> target = (0, '55031181e9bde69634000014', 0, {'SU4': 0.09399, 'L': 0.04445, 'N-1': 0.31915, 'S4': 0.02273, 'N-2': 0.13043}, 'Hirschsprung disease (HSCR) is a multifactorial, non-mendelian disorder in which rare high-penetrance coding sequence mutations in the receptor tyrosine kinase RET contribute to risk in combination with mutations at other genes')
    >>> next(rouge) == target
    True
    >>> target2 = (0, '55031181e9bde69634000014', 1, {'SU4': 0.2, 'L': 0.09639, 'N-1': 0.41379, 'S4': 0.04938, 'N-2': 0.18823}, "In this study, we review the identification of genes and loci involved in the non-syndromic common form and syndromic Mendelian forms of Hirschsprung's disease.")
    >>> next(rouge) == target2
    True
    """
    for qi in range(len(corpus_data)):
        ai = 0
        if type(corpus_data[qi]['ideal_answer']) == list:
            ideal_answers = corpus_data[qi]['ideal_answer']
        else:
            ideal_answers = [corpus_data[qi]['ideal_answer']]
        for answer in ideal_answers:
            modelfilename = os.path.join('..', 'rouge', 'models', 'model_'+str(ai))
            with codecs.open(modelfilename,'w','utf-8') as fout:
                a = '<html>\n<head>\n<title>system</title>\n</head>\n<body bgcolor="white">\n<a name="1">[1]</a> <a href="#1" id=1>{0}</a></body>\n</html>'.format(answer)
                fout.write(a + "\n")
            ai += 1
        if 'snippets' not in corpus_data[qi].keys():
            print("Warning: No snippets in question: %s" % corpus_data[qi]['body'])
            continue
#        for qsnipi in range(len(data[qi]['snippets'])):
#            text = data[qi]['snippets'][qsnipi]['text']
        for (pubmedid,senti,sent) in yield_candidate_text(corpus_data[qi],
                                                          snippets_only):
            #text = data[qi]['snippets'][qsnipi]['text']
            #senti = -1
            #for sent in sent_tokenize(text):
            #    senti += 1
                with open(xml_rouge_filename,'w') as rougef:
                    rougef.write("""<ROUGE-EVAL version="1.0">
 <EVAL ID="1">
 <MODEL-ROOT>
 ../rouge/models
 </MODEL-ROOT>
 <PEER-ROOT>
 ../rouge/summaries
 </PEER-ROOT>
 <INPUT-FORMAT TYPE="SEE" />
 <PEERS>
   <P ID="A">summary_1</P>
 </PEERS>
 <MODELS>
""")
                    for ai in range(len(ideal_answers)):
                        rougef.write(  '<M ID="%i">model_%i</M>\n' % (ai,ai))
                    rougef.write("""</MODELS>
</EVAL>
</ROUGE-EVAL>""")

                peerfilename = os.path.join('..', 'rouge', 'summaries', 'summary_1')
                with codecs.open(peerfilename,'w', 'utf-8') as fout:
                    a = '<html>\n<head>\n<title>system</title>\n</head>\n<body bgcolor="white">\n<a name="1">[1]</a> <a href="#1" id=1>{0}</a></body>\n</html>'.format(sent)
                    fout.write(a + '\n')
                ROUGE_CMD = 'perl ../rouge/RELEASE-1.5.5/ROUGE-1.5.5.pl -e ../rouge/RELEASE-1.5.5/data -c 95 -2 4 -u -x -n 4 -a ' + xml_rouge_filename
                # ROUGE_CMD = 'perl ../rouge/RELEASE-1.5.5/ROUGE-1.5.5.pl -e ../rouge/RELEASE-1.5.5/data -a -n 2 -2 4 -U %s' % (xml_rouge_filename)
                stream = Popen(ROUGE_CMD, shell=True, stdout=PIPE).stdout
                lines = stream.readlines()
                stream.close()
                F = {'N-1':float(lines[3].split()[3]),
                     'N-2':float(lines[7].split()[3]),
                     'L':float(lines[11].split()[3]),
                     'S4':float(lines[15].split()[3]),
                     'SU4':float(lines[19].split()[3])}
#                yield (qi,qsnipi,senti,F,sent)
                yield (qi,pubmedid,senti,F,sent)

def saveRouge(corpus_file, outfile, snippets_only=True):
    "Compute and save the ROUGE scores of the individual snippet sentences"
    corpus_data = loaddata(corpus_file)
    with open(outfile,'w') as f:
        writer = csv.writer(f)
#        writer.writerow(('qid','snipid','sentid','N1','N2','L','S4','SU4','sentence text'))
        writer.writerow(('qid', 'pubmedid', 'sentid', 'N1', 'N2', 'L', 'S4', 'SU4', 'sentence text'))
        progress = progressbar.ProgressBar(max_value=len(corpus_data))
        old_qi = -1
        for (qi, qsnipi, senti, F, sent) in yieldRouge(corpus_data, snippets_only=snippets_only):
            writer.writerow((qi, qsnipi, senti, F['N-1'], F['N-2'], F['L'], F['S4'], F['SU4'], sent))
            if qi != old_qi:
                old_qi = qi
                progress.update(qi+1)


def rouge_to_labels(rougeFile, labels, labelsthreshold, metric=["SU4"]):
    """Convert ROUGE values into classification labels
    >>> labels = rouge_to_labels("rouge_10b.csv", "topn", 3)
    Setting top 3 classification labels
    >>> labels[(0, '55031181e9bde69634000014', 9)]
    False
    >>> labels[(0, '55031181e9bde69634000014', 20)]
    True
    >>> labels[(0, '55031181e9bde69634000014', 3)]
    True
    """
    assert labels in ["topn", "threshold"]

    # Collect ROUGE values
    rouge = dict()
    with codecs.open(rougeFile,'r','utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        index = [header.index(m) for m in metric]
        for line in reader:
            try:
                key = (int(line[0]),line[1],int(line[2]))
            except:
                print("Unexpected error:", sys.exc_info()[0])
                print([l.encode('utf-8') for l in line])
            else:
                rouge[key] = np.mean([float(line[i]) for i in index])

    # Convert ROUGE values into classification labels
    result = dict()
    if labels == "threshold":
        print("Setting classification labels with threshold", labelsthreshold)
        for key, value in rouge.items():
            result[key] = (value >= labelsthreshold)
        qids = set(k[0] for k in rouge)
        # Regardless of threshold, set top ROUGE of every question to True
        for qid in qids:
            qid_items = [(key, value) for key, value in rouge.items() if key[0] == qid]
            qid_items.sort(key = lambda k: k[1])
            result[qid_items[-1][0]] = True
    else:
        print("Setting top", labelsthreshold, "classification labels")
        qids = set(k[0] for k in rouge)
        for qid in qids:
            qid_items = [(key, value) for key, value in rouge.items() if key[0] == qid]
            qid_items.sort(key = lambda k: k[1])
            for k, v in qid_items[-labelsthreshold:]:
                result[k] = True
            for k, v in qid_items[:-labelsthreshold]:
                result[k] = False
    return result

class BaseClassification:
    """A base classification to be inherited"""
    def __init__(self, corpusFile, rougeFile, metric=['SU4'],
                 rouge_labels=True, labels="topn", labelsthreshold=5):
        """Initialise the classification system."""
        print("Reading data from %s and %s" % (corpusFile, rougeFile))
        self.data = loaddata(corpusFile)
        if not rouge_labels:
            #convert rouge -> label
            self.labels = rouge_to_labels(rougeFile, labels, labelsthreshold, metric=metric)
        else:
            #leave as rouge scores
            self.labels = dict()
            with codecs.open(rougeFile, encoding='utf-8') as csv_file:
                reader = csv.DictReader(csv_file)
                lineno = 0
                num_errors = 0
                for line in reader:
                    lineno += 1
                    try:
                        key = (int(line['qid']), line['pubmedid'], int(line['sentid']))
                    except:
                        num_errors += 1
                        print("Unexpected error:", sys.exc_info()[0])
                        print("%i %s" % (lineno, str(line).encode('utf-8')))
                    else:
                        self.labels[key] = np.mean([float(line[m]) for m in metric])
                if num_errors > 0:
                    print("%i data items were ignored out of %i because of errors" % (num_errors, lineno))

    def _collect_data_(self, indices, testindices=None):
        """Collect the data given the question indices"""
        print("Collecting data")
        with Pool() as pool:
            collected = pool.map(partial(collect_one_item,
                                         indices=indices,
                                         testindices=testindices,
                                         data=self.data,
                                         labels=self.labels),
                                 enumerate(self.data))
        all_candidates_questions = {'main':[], 'test':[]}
        all_candidates_sentences = {'main':[], 'test':[]}
        all_candidates_sentences_ids = {'main':[], 'test':[]}
        all_labels = {'main':[], 'test':[]}
        for c in collected:
            if c == None:
                continue
            partition, labels_data, candidates_questions, candidates_sentences, candidates_sentences_ids = c
            all_candidates_questions[partition] += candidates_questions
            all_candidates_sentences[partition] += candidates_sentences
            all_candidates_sentences_ids[partition] += candidates_sentences_ids
            all_labels[partition] += labels_data

        print("End collecting data")
        return all_labels, all_candidates_questions, all_candidates_sentences, all_candidates_sentences_ids

class Classification(BaseClassification):
    """A classification system"""
    def __init__(self, corpusFile, rougeFile, metric=['SU4'],
                 rouge_labels=True, labels="topn", labelsthreshold=5,
                 nb_epoch=3, verbose=2,
                 classification_type="BERT",
                 hidden_layer=0,
                 dropout=0.5,
                 batch_size=128,
                 finetune_model=False):
        """Initialise the classification system."""
        BaseClassification.__init__(self, corpusFile, rougeFile, metric=metric,
                                    rouge_labels=rouge_labels, labels=labels, 
                                    labelsthreshold=labelsthreshold)
        self.nb_epoch = nb_epoch
        self.verbose = verbose
        self.dropout = dropout
        self.batch_size = batch_size
        self.finetune_model = finetune_model
        self.classification_type = classification_type
        self.hidden_layer_size = hidden_layer


        self.nnc = None

    def extractfeatures(self, questions, candidates_sentences):
        """ Return the features"""
        assert len(questions) == len(candidates_sentences)

        return ([self.nnc.cleantext(sentence) for sentence in candidates_sentences],
                [self.nnc.cleantext(question) for question in questions])

    def create_classifier_instance(self):
        print("Creating training instance")

        if self.classification_type == "BERT":
            self.nnc = nnc.BERT(hidden_layer_size=self.hidden_layer_size,
                                 batch_size=self.batch_size,
                                 positions=True,
                                 dropout_rate=self.dropout,
                                 finetune_model=self.finetune_model)
        elif self.classification_type == "BioBERT":
            self.nnc = nnc.BioBERT(hidden_layer_size=self.hidden_layer_size,
                                   batch_size=self.batch_size,
                                   positions=True,
                                   dropout_rate=self.dropout,
                                   finetune_model=self.finetune_model)
        elif self.classification_type == "DistilBERT":
            self.nnc = nnc.DistilBERT(hidden_layer_size=self.hidden_layer_size,
                                      batch_size=self.batch_size,
                                      positions=True,
                                      dropout_rate=self.dropout,
                                      finetune_model=self.finetune_model)
        elif self.classification_type == "ALBERT":
            self.nnc = nnc.ALBERT(hidden_layer_size=self.hidden_layer_size,
                                  batch_size=self.batch_size,
                                  positions=True,
                                  dropout_rate=self.dropout,
                                  finetune_model=self.finetune_model)
        elif self.classification_type == "ALBERT_squad2":
            self.nnc = nnc.ALBERT_squad2(hidden_layer_size=self.hidden_layer_size,
                                  batch_size=self.batch_size,
                                  positions=True,
                                  dropout_rate=self.dropout,
                                  finetune_model=self.finetune_model)
        elif self.classification_type == "QA_ALBERT":
            self.nnc = nnc.QA_ALBERT(hidden_layer_size=self.hidden_layer_size,
                                  batch_size=self.batch_size,
                                  positions=True,
                                  dropout_rate=self.dropout,
                                  finetune_model=self.finetune_model)
        elif self.classification_type == "Pos":
            self.nnc = nnc.Pos(hidden_layer_size=self.hidden_layer_size,
                            batch_size=self.batch_size,
                            positions=True,
                            dropout_rate=self.dropout)

        print("%s with epochs=%i and batch size=%i" % (self.nnc.name(), 
                                                       self.nb_epoch,
                                                       self.batch_size))



    def train(self, indices, testindices=None, foldnumber=0, restore_model=False,
              savepath=None,
              save_test_predictions=None):
        """Train the classifier given the question indices"""

        self.create_classifier_instance()

        print("Gathering training data")
        if savepath is None:
            savepath="savedmodels/%s_%i" % (self.nnc.name(), foldnumber)
        all_labels, candidates_questions, candidates_sentences, candidates_sentences_ids = \
        self._collect_data_(indices, testindices)

        features = self.extractfeatures(candidates_questions['main'],
                                        candidates_sentences['main'])

        print("Training %s" % self.nnc.name())
        if testindices == None:
            validation_data = None
        else:
            features_test = self.extractfeatures(candidates_questions['test'],
                                                 candidates_sentences['test'])
            validation_data = (features_test[0], features_test[1],
                               [[r] for r in all_labels['test']],
                               [[cid] for cid in candidates_sentences_ids['test']])
        loss_history = self.nnc.fit(features[0], features[1],
                                    np.array([[r] for r in all_labels['main']]),
                                    X_positions=np.array([[cid] for cid in candidates_sentences_ids['main']]),
                                    validation_data = validation_data,
                                    nb_epoch=self.nb_epoch,
                                    verbose=self.verbose,
                                    savepath=savepath,
                                    restore_model=restore_model)        
        if save_test_predictions:
            predictions_test = self.nnc.predict(features_test[0],
                                                features_test[1],
                                                X_positions=np.array([[cid] for cid in candidates_sentences_ids['test']]))
            predictions_test = [p[0] for p in predictions_test]
            print("Saving predictions in %s" % save_test_predictions)
            with open(save_test_predictions, "w") as f:
                writer = csv.DictWriter(f, fieldnames=["id", "target", "prediction"])
                writer.writeheader()
                for i, p in enumerate(predictions_test):
                    writer.writerow({"id": i,
                                     "target": all_labels['test'][i],
                                     "prediction": p})
                print("Predictions saved")

        if restore_model:
            return

        if testindices:
            return loss_history['loss'][-1], loss_history['val_loss'][-1]
        else:
            return loss_history['loss'][-1]

    def save(self, savepath):
        self.nnc.save(savepath)

    def load(self, loadpath):
        self.create_classifier_instance()
        self.nnc.load(loadpath)

    def test(self, indices):
        """Test the classifier given the question indices"""
        print("Gathering test data")
        all_labels, candidates_questions, candidates_sentences, candidates_sentences_ids = \
        self._collect_data_(indices)

        features = self.extractfeatures(candidates_questions['main'],
                                        candidates_sentences['main'])

        print("Testing NNC")
        loss = self.nnc.test(features[0], 
                              features[1],
                              [[r] for r in all_labels['main']],
                              X_positions=[[cid] for cid in candidates_sentences_ids['main']])
        print("Loss = %f" % loss)
        return loss

    def answersummaries(self, questions_and_candidates, beamwidth=0):
        if beamwidth > 0:
            print("Beam width is", beamwidth)
            return answersummaries(questions_and_candidates, self.extractfeatures, self.nnc.predict, beamwidth)
        else:
            return answersummaries(questions_and_candidates, self.extractfeatures, self.nnc.predict)

    def answersummary(self, question, candidates_sentences,
                      n=3, qindex=None):
        """Return a summary that answers the question

        qindex is not used but needed for compatibility with oracle"""
        return self.answersummaries((question, candidates_sentences, n))

def evaluate_one(di, dataset, testindices, nanswers, rougepath):
    """Evaluate one question"""
    if di not in testindices:
        return None
    question = dataset[di]['body']
    if 'snippets' not in dataset[di].keys():
        return None
    candidates = [(sent, sentid) for (pubmedid, sentid, sent) in yield_candidate_text(dataset[di])]
    if len(candidates) == 0:
        # print("Warning: No text to summarise; ignoring this text")
        return None

    if type(nanswers) == dict:
        n = nanswers[dataset[di]['type']]
    else:
        n = nanswers
    rouge_text = """<EVAL ID="%i">
 <MODEL-ROOT>
 %s/models
 </MODEL-ROOT>
 <PEER-ROOT>
 %s/summaries
 </PEER-ROOT>
 <INPUT-FORMAT TYPE="SEE">
 </INPUT-FORMAT>
""" % (di, rougepath, rougepath)
    rouge_text += """ <PEERS>
  <P ID="A">summary%i.txt</P>
 </PEERS>
 <MODELS>
""" % (di)

    if type(dataset[di]['ideal_answer']) == list:
        ideal_answers = dataset[di]['ideal_answer']
    else:
        ideal_answers = [dataset[di]['ideal_answer']]

    for j in range(len(ideal_answers)):
        rouge_text += '  <M ID="%i">ideal_answer%i_%i.txt</M>\n' % (j,di,j)
        with codecs.open(rougepath + '/models/ideal_answer%i_%i.txt' % (di,j),
                         'w', 'utf-8') as fout:
            a = '<html>\n<head>\n<title>system</title>\n</head>\n<body bgcolor="white">\n<a name="1">[1]</a> <a href="#1" id=1>{0}</body>\n</html>'.format(ideal_answers[j])
            fout.write(a+'\n')
    rouge_text += """ </MODELS>
</EVAL>
"""
    target = {'id': dataset[di]['id'],
              'ideal_answer': ideal_answers,
              'exact_answer': ""}
    return rouge_text, di, (question, candidates, n), target

def evaluate(classificationClassInstance, rougeFilename="rouge.xml", nanswers=3,
             tmppath='', load_models=False, small_data=False, fold=0, data_indices=None):
    """Evaluate a classification-based summariser

    nanswers is the number of answers. If it is a dictionary, then the keys indicate the question type, e.g.
    nanswers = {"summary": 6,
                "factoid": 2,
                "yesno": 2,
                "list": 3}
"""
    if tmppath == '':
        modelspath = 'saved_models_Similarities'
        rougepath = '../rouge'
        crossvalidationpath = 'crossvalidation'
    else:
        modelspath = tmppath + '/saved_models'
        rougepath = tmppath + '/rouge'
        crossvalidationpath = tmppath + '/crossvalidation'
        rougeFilename = rougepath + "/" + rougeFilename
        if not os.path.exists(rougepath):
            os.mkdir(rougepath)
        for f in glob.glob(rougepath + '/*'):
            if os.path.isfile(f):
                os.remove(f)
            elif os.path.isdir(f):
                shutil.rmtree(f)
            else:
                print("Warning: %f is neither a file nor a directory" % (f))
        os.mkdir(rougepath + '/models')
        os.mkdir(rougepath + '/summaries')
        if not os.path.exists(crossvalidationpath):
            os.mkdir(crossvalidationpath)

    dataset = classificationClassInstance.data

    if data_indices:
        indices = data_indices
    else:
        indices = [i for i in range(len(dataset))
                   #if dataset[i]['type'] == 'summary'
                   #if dataset[i]['type'] == 'factoid'
                   #if dataset[i]['type'] == 'yesno'
                   #if dataset[i]['type'] == 'list'
                  ]
    if small_data:
        indices = indices[:100]

    random.seed(1234)
    random.shuffle(indices)

    rouge_results = []
    rouge_results_P = []
    rouge_results_R = []
    the_fold = 0
    kf = KFold(n_splits=10)
    for (traini, testi) in kf.split(indices):
        the_fold += 1

        if fold > 0 and the_fold != fold:
            continue

        if small_data and the_fold > 2:
           break

        print("Cross-validation Fold %i" % the_fold)
        trainindices = [indices[i] for i in traini]
        testindices = [indices[i] for i in testi]

        save_test_predictions = crossvalidationpath + "/test_results_%i.csv" % the_fold
        (trainloss,testloss) = classificationClassInstance.train(trainindices,
                                                             testindices,
                                                             foldnumber=the_fold,
                                                             restore_model=load_models,
                                                             savepath="%s/saved_model_%i" % (modelspath, the_fold),
                                                             save_test_predictions=save_test_predictions)

        for f in glob.glob(rougepath+'/models/*')+glob.glob(rougepath+'/summaries/*'):
            os.remove(f)

        with open(rougeFilename,'w') as frouge:
           print("Collecting evaluation results")
           frouge.write('<ROUGE-EVAL version="1.0">\n')
           #with Pool() as pool:
           #    evaluation_data = \
           #       pool.map(partial(evaluate_one,
           #                        dataset=dataset,
           #                        testindices=testindices,
           #                        nanswers=nanswers,
           #                        rougepath=rougepath),
           #                range(len(dataset)))

           evaluation_data = [evaluate_one(i,
                                           dataset=dataset,
                                           testindices=testindices,
                                           nanswers=nanswers,
                                           rougepath=rougepath)
                              for i in range(len(dataset))]


           summaries = classificationClassInstance.answersummaries([e[2] for e in evaluation_data if e != None])

           eval_test_system = []
           eval_test_target = []
           for data_item in evaluation_data:
               if data_item == None:
                   continue
               rouge_item, di, system_item, target_item = data_item
               summary = summaries.pop(0)
               #print(di)
               with codecs.open(rougepath+'/summaries/summary%i.txt' % (di),
                               'w', 'utf-8') as fout:
                   a = '<html>\n<head>\n<title>system</title>\n</head>\n<body bgcolor="white">\n<a name="1">[1]</a> <a href="#1" id=1>{0}</body>\n</html>'.format(" ".join(summary))
                   fout.write(a+'\n')
                   # fout.write('\n'.join([s for s in summary])+'\n')

               frouge.write(rouge_item)
               system_item = {'id': dataset[di]['id'],
                              'ideal_answer': " ".join(summary),
                              'exact_answer': ""}

               eval_test_system.append(system_item)
               eval_test_target.append(target_item)

           assert len(summaries) == 0

           frouge.write('</ROUGE-EVAL>\n')

        json_summaries_file = crossvalidationpath + "/crossvalidation_%i_summaries.json" % the_fold
        print("Saving summaries in file %s" % json_summaries_file)
        with open(json_summaries_file,'w') as fcv:
            fcv.write(json.dumps({'questions': eval_test_system}, indent=2))
        json_gold_file = crossvalidationpath + "/crossvalidation_%i_gold.json" % the_fold
        print("Saving gold data in file %s" % json_gold_file)
        with open(json_gold_file,'w') as fcv:
            fcv.write(json.dumps({'questions': eval_test_target}, indent=2))

#        print("Calling ROUGE", rougeFilename)
        ROUGE_CMD = 'perl ../rouge/RELEASE-1.5.5/ROUGE-1.5.5.pl -e ' \
            + '../rouge/RELEASE-1.5.5/data -c 95 -2 4 -u -x -n 4 -a ' \
            + rougeFilename
        print("Calling ROUGE", ROUGE_CMD)
        stream = Popen(ROUGE_CMD, shell=True, stdout=PIPE).stdout
        lines = stream.readlines()
        stream.close()
        for l in lines:
            print(l.decode('ascii').strip())
        print()

        F = {'N-1':float(lines[3].split()[3]),
             'N-2':float(lines[7].split()[3]),
             'L':float(lines[11].split()[3]),
             'S4':float(lines[15].split()[3]),
             'SU4':float(lines[19].split()[3]),
             'trainloss':trainloss,
             'testloss':testloss}
        P = {'N-1':float(lines[2].split()[3]),
             'N-2':float(lines[6].split()[3]),
             'L':float(lines[10].split()[3]),
             'S4':float(lines[14].split()[3]),
             'SU4':float(lines[18].split()[3]),
             'trainloss':trainloss,
             'testloss':testloss}
        R = {'N-1':float(lines[1].split()[3]),
             'N-2':float(lines[5].split()[3]),
             'L':float(lines[9].split()[3]),
             'S4':float(lines[15].split()[3]),
             'SU4':float(lines[17].split()[3]),
             'trainloss':trainloss,
             'testloss':testloss}
        rouge_results.append(F)
        rouge_results_P.append(P)
        rouge_results_R.append(R)

        print("F N-2: %1.5f SU4: %1.5f TrainLoss: %1.5f TestLoss: %1.5f" % (
               F['N-2'], F['SU4'], F['trainloss'], F['testloss']
        ))
        print("P N-2: %1.5f SU4: %1.5f TrainLoss: %1.5f TestLoss: %1.5f" % (
               P['N-2'], P['SU4'], P['trainloss'], P['testloss']
        ))
        print("R N-2: %1.5f SU4: %1.5f TrainLoss: %1.5f TestLoss: %1.5f" % (
               R['N-2'], R['SU4'], R['trainloss'], R['testloss']
        ))


    print("%5s %7s %7s %7s %7s" % ('', 'N-2', 'SU4', 'TrainLoss', 'TestLoss'))
    for i in range(len(rouge_results)):
        print("%5i %1.5f %1.5f %1.5f %1.5f" % (i+1,rouge_results[i]['N-2'],rouge_results[i]['SU4'],
                                       rouge_results[i]['trainloss'],rouge_results[i]['testloss']))
    mean_N2 = np.average([rouge_results[i]['N-2']
                          for i in range(len(rouge_results))])
    mean_SU4 = np.average([rouge_results[i]['SU4']
                           for i in range(len(rouge_results))])
    mean_N2_P = np.average([rouge_results_P[i]['N-2']
                          for i in range(len(rouge_results_P))])
    mean_SU4_P = np.average([rouge_results_P[i]['SU4']
                           for i in range(len(rouge_results_P))])
    mean_N2_R = np.average([rouge_results_R[i]['N-2']
                          for i in range(len(rouge_results_R))])
    mean_SU4_R = np.average([rouge_results_R[i]['SU4']
                           for i in range(len(rouge_results_R))])
    mean_Trainloss = np.average([rouge_results[i]['trainloss']
                                 for i in range(len(rouge_results))])
    mean_Testloss = np.average([rouge_results[i]['testloss']
                                for i in range(len(rouge_results))])
    print("%5s %1.5f %1.5f %1.5f %1.5f" % ("mean",mean_N2,mean_SU4,mean_Trainloss,mean_Testloss))
    stdev_N2 = np.std([rouge_results[i]['N-2']
                       for i in range(len(rouge_results))])
    stdev_SU4 = np.std([rouge_results[i]['SU4']
                        for i in range(len(rouge_results))])
    stdev_N2_P = np.std([rouge_results_P[i]['N-2']
                       for i in range(len(rouge_results_P))])
    stdev_SU4_P = np.std([rouge_results_P[i]['SU4']
                        for i in range(len(rouge_results_P))])
    stdev_N2_R = np.std([rouge_results_R[i]['N-2']
                       for i in range(len(rouge_results_R))])
    stdev_SU4_R = np.std([rouge_results_R[i]['SU4']
                        for i in range(len(rouge_results_R))])
    stdev_Trainloss = np.std([rouge_results[i]['trainloss']
                              for i in range(len(rouge_results))])
    stdev_Testloss = np.std([rouge_results[i]['testloss']
                             for i in range(len(rouge_results))])
    print("%5s %1.5f %1.5f %1.5f %1.5f" % ("stdev",stdev_N2,stdev_SU4,stdev_Trainloss,stdev_Testloss))
    print()
    return mean_SU4, stdev_SU4, mean_SU4_P, stdev_SU4_P, mean_SU4_R, stdev_SU4_R, mean_Testloss, stdev_Testloss

if __name__ == "__main__":
    import doctest
    doctest.testmod()

    #import sys
    #saveRouge("training10b.json", "rouge_10b.csv")
    #bioasq_train(small_data=True, model_type='distilbert')
    #bioasq_train(model_type='bert')
    #bioasq_train(model_type='biobert')
    #bioasq_train(model_type='distilbert') #, small_data=True)
    #bioasq_train(model_type='albert')
    #bioasq_train(model_type='qaalbert', small_data=True)
    #print ("SAVED MODEL - NOW TRY RUNNING IT")
    #bioasq_run(model_type='bert', output_filename='bioasq-out-bert.json')
    #bioasq_run(model_type='biobert', output_filename='bioasq-out-biobert.json')
    #bioasq_run(model_type='distilbert', output_filename='bioasq-out-distilbert.json')
    #bioasq_run(model_type='qaalbert', output_filename='bioasq-out-qaalbert.json')
    #sys.exit()

    import argparse
    import time
    import socket
    import mlflow

    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--nb_epoch', type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument('-v', '--verbose', type=int, default=1,
                        help="Verbosity level")
    parser.add_argument('-t', '--classification_type',
                        choices=("BERT", "BioBERT", "DistilBERT", "ALBERT", "ALBERT_squad2", "QA_ALBERT", "Pos"),
                        default="DistilBERT",
                        help="Type of classification")
    parser.add_argument('-S', '--small', action="store_true",
                        help='Run on a small subset of the data')
    parser.add_argument('-T', '--truncate_training', action="store_true",
                        help="Use part of the training data")
    parser.add_argument('-l', '--load', action="store_true",
                        help='load pre-trained model')
    parser.add_argument('-d', '--hidden_layer', type=int, default=50,
                        help="Size of the hidden layer (0 if there is no hidden layer)")
    parser.add_argument('-r', '--dropout', type=float, default=0.6,
                        help="Keep probability for the dropout layers")
    parser.add_argument('-a', '--tmppath', default='',
                        help="Path for temporary data and files")
    parser.add_argument('-n', '--rouge_labels', default=False, action='store_true',
                        help="Use raw rouge scores as labels. If not specified labels are converted to True/False categories")
    parser.add_argument('-F', '--finetune_model', default=False, action='store_true',
                        help="Finetune the model. If not specified the model is frozen and only classification layers are finetuned")
    parser.add_argument('-s', '--batch_size', type=int, default=32,
                        help="Batch size for gradient descent")
    parser.add_argument("-f", "--fold", type=int, default=0,
                        help="Use only the specified fold (0 for all folds)")
    parser.add_argument('-x', '--mlflow_experiment_id', type=int, default=0,
                        help="MLFlow experiment ID")

    args = parser.parse_args()

    print("rouge_labels: %s" % args.rouge_labels)

    with mlflow.start_run(experiment_id=args.mlflow_experiment_id):
        classifier = Classification(#'BioASQ-training8b.json',
                                    #'rouge_8b.csv',
                                    'training10b.json',
                                    'rouge_10b.csv',
                                    rouge_labels=args.rouge_labels,
                                    nb_epoch=args.nb_epoch,
                                    verbose=args.verbose,
                                    classification_type=args.classification_type,
                                    hidden_layer=args.hidden_layer,
                                    dropout=args.dropout,
                                    batch_size=args.batch_size,
                                    finetune_model=args.finetune_model)

        if args.truncate_training:
            # Use only the last 50% of the training data
            size_data = len(classifier.data)
            percent = 50
            #data_indices = list(range(0,size_data*percent/100)))
            data_indices = list(range(int((100-percent)*size_data/100), size_data))
            mlflow.log_param("Percent training data", len(data_indices)*100/size_data)
        else:
            data_indices = list(range(len(classifier.data)))

        mlflow.log_param("Number epochs", args.nb_epoch)
        mlflow.log_param("Dropout rate", args.dropout)
        mlflow.log_param("Fold", args.fold)
        if args.small:
            mlflow.log_param("Small data", True)


        mean_SU4, stdev_SU4, mean_SU4_P, stdev_SU4_P, mean_SU4_R, stdev_SU4_R, mean_Testloss, stdev_Testloss = \
                                evaluate(classifier,
                                        nanswers={"summary": 6,
                                                    "factoid": 2,
                                                    "yesno": 2,
                                                    "list": 3},
                                        tmppath=args.tmppath,
                                        load_models=args.load,
                                        small_data=args.small,
                                        fold = args.fold,
                                        data_indices = data_indices)
        end_time = time.time()
        elapsed = time.strftime("%X", time.gmtime(end_time - start_time))

        mlflow.log_param("model", classifier.nnc.name())
        mlflow.log_metric("Mean SU4", mean_SU4)
        mlflow.log_metric("Std SU4", stdev_SU4)

    print("Time elapsed: %s" % (elapsed))
    print("| Type | Fold | Epochs | Dropout | meanSU4 | stdevSU4 | meanSU4_P | stdevSU4_P | meanSU4_R | stdevSU4_R | meanTestLoss | stdevTestLoss | Time | Hostname |")
    print("| %s | %i | %i | %f | %f | %f | %f | %f | %f | %f | %f | %f | %s | %s |" % \
               (classifier.nnc.name(),
                args.fold,
                args.nb_epoch,
                args.dropout,
                mean_SU4,
                stdev_SU4,
                mean_SU4_P,
                stdev_SU4_P,
                mean_SU4_R,
                stdev_SU4_R,
                mean_Testloss,
                stdev_Testloss,
                elapsed,
                socket.gethostname()))

