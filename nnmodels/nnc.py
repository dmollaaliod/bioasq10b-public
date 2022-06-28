"NN models for classification"
import numpy as np
from multiprocessing import Pool
from functools import partial

from transformers import BertModel, BertTokenizer
from transformers import DistilBertModel, DistilBertTokenizer
from transformers import AlbertModel, AlbertTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import progressbar

# import mlflow
# import logging

#logging.basicConfig(level=logging.WARN)
#logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using", device, "device")

def sentence_to_qbert_id(X_Q, tokenizer, max_length):
    result = tokenizer(X_Q[0], X_Q[1], add_special_tokens=True, 
                                           max_length=max_length,
                                           padding='max_length',
                                           truncation=True)

    if 'token_type_ids' not in result:
        # This part is needed because DistilBERT doesn't use token_type_ids
        segment_start = result['input_ids'].index(102) + 1
        if 0 in result['attention_mask']:
            segment_end = result['attention_mask'].index(0) - 1
        else:
            segment_end = len(result['attention_mask']) - 1
        #print(result)    
        #print(result['input_ids'][segment_start:segment_end])

        result['token_type_ids'] = [0] * segment_start + [1] * (segment_end - segment_start) + [0] * (len(result['input_ids']) - segment_end)
        assert len(result['token_type_ids']) == len(result['input_ids'])


    return result
    

def parallel_sentences_to_qbert_ids(batch_X, batch_Q, 
                                   sentence_length,
                                   tokenizer):
    "Convert the text to indices and pad-truncate to the maximum number of words"
    assert len(batch_X) == len(batch_Q)
    with Pool() as pool:
        result = pool.map(partial(sentence_to_qbert_id,
                                tokenizer=tokenizer,
                                max_length=sentence_length),
                        zip(batch_Q, batch_X))


    return {'input_ids': [x['input_ids'] for x in result],
            'attention_mask': [x['attention_mask'] for x in result],
            'token_type_ids': [x['token_type_ids'] for x in result]}

class BERT(nn.Module):
    """Simplest possible BERT classifier"""
    def __init__(self, batch_size=32, hidden_layer_size=0, dropout_rate=0.5, 
                 positions=False, finetune_model=False):
        super(BERT, self).__init__()
        self.batch_size = batch_size
        self.sentence_length = 250 # as observed in exploratory file bert-exploration.ipynb
        self.hidden_layer_size = hidden_layer_size
        self.tokenizer = self.load_tokenizer()
        self.cleantext = lambda t: t
        self.positions = positions
        self.finetune = finetune_model
        self.dropout_rate = dropout_rate

        # Model layers   
        bert_config, self.bert_layer = self.load_bert_model()
        for param in self.bert_layer.parameters():
            param.requires_grad = False

        self.bert_hidden_size = bert_config['hidden_size']

        if self.positions:
            bert_size = self.bert_hidden_size + 1

        else:
            bert_size = self.bert_hidden_size

        if hidden_layer_size > 0:
            self.hidden_layer = nn.Linear(bert_size, 
                                          hidden_layer_size)
            output_size = hidden_layer_size
        else:
            output_size = bert_size
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        self.output_layer = nn.Linear(output_size, 1)

        self.to(device)

        print("Number of parameters:",
              sum(p.numel() for p in self.parameters()))
        print("Number of trainable parameters:",
              sum(p.numel() for p in self.parameters() if p.requires_grad))

    def save(self, savepath):
        torch.save(self.state_dict(), savepath)

    def load(self, loadpath):
        self.load_state_dict(torch.load(loadpath))

    def forward(self, input_word_ids, input_masks, input_segments, avg_mask, positions):

        # BERT
        if self.name().startswith('DistilBERT'):
            #print("We are in DistilBERT")
            bert = self.bert_layer(input_ids=input_word_ids, 
                                   attention_mask=input_masks)
        else:
            #print("We are NOT in DistilBERT")
            bert = self.bert_layer(input_ids=input_word_ids, 
                                   attention_mask=input_masks, 
                                   token_type_ids=input_segments)

        # Average using avg_mask
        mask = avg_mask.unsqueeze(2).repeat(1, 1, self.bert_hidden_size)
        inputs_sum = torch.sum(bert.last_hidden_state * mask, 1)
        pooling = inputs_sum / torch.sum(mask, 1)

        # Sentence positions
        if self.positions:
            all_inputs = torch.cat([pooling, positions], 1)

        else:
            all_inputs = pooling

        # Hidden layer
        if self.hidden_layer_size > 0:
            hidden = F.relu(self.hidden_layer(all_inputs))
        else:
            hidden = all_inputs
        dropout = self.dropout_layer(hidden)

        # Final outputs
        outputs = torch.sigmoid(self.output_layer(dropout))

        return(outputs)

    def load_tokenizer(self):
        return BertTokenizer.from_pretrained('bert-base-uncased')

    def load_bert_model(self):
        return {'hidden_size': 768}, BertModel.from_pretrained('bert-base-uncased')
 
    def name(self):
        if self.positions:
            str_positions = "+pos"
        else:
            str_positions = ""
        if self.hidden_layer_size == 0:
            str_hidden = ""
        else:
            str_hidden = "-relu(%i)" % self.hidden_layer_size
        if self.finetune:
            str_finetune = "(finetuned)"
        else:
            str_finetune = "(frozen)"
        return "BERT%s%s%s" % (str_finetune, str_positions, str_hidden)

    def fit(self, X_train, Q_train, Y_train,
            X_positions = [],
            validation_data=None,
            verbose=2, nb_epoch=3,
            savepath=None, restore_model=False):

        # Training loop
        X = parallel_sentences_to_qbert_ids(X_train, Q_train, self.sentence_length, self.tokenizer)

        return self.__fit__(X, X_positions, Y_train,
                            validation_data,
                            verbose, nb_epoch, None, savepath)

    def __fit__(self, X, X_positions, Y,
                validation_data,
                verbose, nb_epoch, learning_rate,
                savepath=None):

        criterion = nn.BCELoss()
        if learning_rate:
            optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        else:
            optimizer = optim.Adam(self.parameters())

        history = {'loss':[], 'val_loss':[]}

        print("Batch size:", self.batch_size)
        print("Input data size:", len(X['input_ids']))

        # mlflow.log_param("model", self.name())
        # mlflow.log_param("number_epochs", nb_epoch)
        # mlflow.log_param("dropout_rate", self.dropout_rate)
        self.train()
        for epoch in range(nb_epoch):
            if self.finetune and epoch == 1:
                print("Unfreezing the model")
                for param in self.bert_layer.parameters():
                    param.requires_grad = True
                for param in list(self.bert_layer.embeddings.parameters()):
                    param.requires_grad = False

                print("Number of parameters:",
                sum(p.numel() for p in self.parameters()))
                print("Number of trainable parameters:",
                sum(p.numel() for p in self.parameters() if p.requires_grad))


            #print("Entering epoch", epoch)
            # running_loss = 0.0
            epoch_loss = 0.0
            batch_count = 0
            progress = progressbar.ProgressBar(max_value=len(X['input_ids']) // self.batch_size,
                                            variables={'loss': '--'},
                                            suffix=' Loss: {variables.loss}',
                                            redirect_stdout=True).start()
            for batch in range(len(X['input_ids']) // self.batch_size):
                f = self.batch_size*batch
                t = self.batch_size*(batch+1)
                input_word_ids = torch.tensor(X['input_ids'][f:t], device=device)
                input_masks = torch.tensor(X['attention_mask'][f:t], device=device)
                input_segments = torch.tensor(X['token_type_ids'][f:t], device=device)
                input_positions = torch.tensor(X_positions[f:t], device=device).float()
                avg_mask = torch.tensor(X['token_type_ids'][f:t], device=device)
                targets = torch.tensor(Y[f:t], device=device)#.unsqueeze(1)

                #print("Size of inputs:", input_word_ids.size())

                optimizer.zero_grad()
                outputs = self(input_word_ids, input_masks, input_segments, avg_mask, input_positions)
                #print("Size of outputs:", outputs.size())
                #print("Size of targets:", targets.size())
                loss = criterion(outputs.float(), targets.float())
                loss.backward()
                optimizer.step()

                # running_loss += loss.item()
                epoch_loss += loss.item()
                batch_count += 1

                # modulo = 1 # Report after this number of batches
                # if batch % modulo == modulo - 1:
                #     print('[%d, %5d] loss: %.3f' %
                #           (epoch + 1, batch + 1, running_loss / modulo))
                #     running_loss = 0.0
                progress.update(batch+1, loss="%.4f" % loss.item())

            history['loss'].append(epoch_loss / batch_count)

            if validation_data:
                history['val_loss'].append(self.test(*validation_data)['loss'])
                print("Epoch", epoch+1, "loss:", history['loss'][-1], 
                    "validation loss:", history['val_loss'][-1])
            else:
                print("Epoch", epoch+1, "loss:", history['loss'][-1])
        # mlflow.log_metric("training_loss", history['loss'][-1])
        # if validation_data:
        #     mlflow.log_metric("validation_loss", history['val_loss'][-1])

        return history


    def predict(self, X_topredict, Q_topredict, X_positions=[]):
        X = parallel_sentences_to_qbert_ids(X_topredict, Q_topredict, self.sentence_length, self.tokenizer)
        # input_word_ids = torch.tensor(X['input_ids'], device=device)
        # input_masks = torch.tensor(X['attention_mask'], device=device)
        # input_segments = torch.tensor(X['token_type_ids'], device=device)
        # input_positions = torch.tensor(X_positions, device=device).float()
        # avg_mask = torch.tensor(X['token_type_ids'], device=device)

        progress = progressbar.ProgressBar(max_value=len(X['input_ids']) // self.batch_size,
                                               redirect_stdout=True).start()

        result = []

        self.eval()
        with torch.no_grad():
            for batch in range(len(X['input_ids']) // self.batch_size):
                f = self.batch_size*batch
                t = self.batch_size*(batch+1)
                input_word_ids = torch.tensor(X['input_ids'][f:t], device=device)
                input_masks = torch.tensor(X['attention_mask'][f:t], device=device)
                input_segments = torch.tensor(X['token_type_ids'][f:t], device=device)
                input_positions = torch.tensor(X_positions[f:t], device=device).float()

                avg_mask = torch.tensor(X['token_type_ids'][f:t], device=device)

                result += self(input_word_ids, input_masks, input_segments, avg_mask, input_positions)

                progress.update(batch+1)

            if len(X['input_ids']) % self.batch_size != 0:
                # Remaining data 
                f =  len(X['input_ids']) // self.batch_size * self.batch_size
                input_word_ids = torch.tensor(X['input_ids'][f:], device=device)
                input_masks = torch.tensor(X['attention_mask'][f:], device=device)
                input_segments = torch.tensor(X['token_type_ids'][f:], device=device)
                input_positions = torch.tensor(X_positions[f:], device=device).float()

                avg_mask = torch.tensor(X['token_type_ids'][f:], device=device)

                result += self(input_word_ids, input_masks, input_segments, avg_mask, input_positions)

        return result

    def test(self, X_test, Q_test, Y_test, X_positions=[]):
        X = parallel_sentences_to_qbert_ids(X_test, Q_test, self.sentence_length, self.tokenizer)
        return self.__test__(X, Y_test, X_positions)

    def __test__(self, X,
                       Y,
                       X_positions=[]):

        criterion = nn.BCELoss()
        test_loss = 0.0
        loss_count = 0
        progress = progressbar.ProgressBar(max_value=len(X['input_ids']) // self.batch_size,
                                               variables={'loss': '--'},
                                               suffix=' Loss: {variables.loss}',
                                               redirect_stdout=True).start()

        self.eval()
        with torch.no_grad():
            for batch in range(len(X['input_ids']) // self.batch_size):
                f = self.batch_size*batch
                t = self.batch_size*(batch+1)
                input_word_ids = torch.tensor(X['input_ids'][f:t], device=device)
                input_masks = torch.tensor(X['attention_mask'][f:t], device=device)
                input_segments = torch.tensor(X['token_type_ids'][f:t], device=device)
                input_positions = torch.tensor(X_positions[f:t], device=device).float()

                avg_mask = torch.tensor(X['token_type_ids'][f:t], device=device)
                targets = torch.tensor(Y[f:t], device=device)#.unsqueeze(1)

                outputs = self(input_word_ids, input_masks, input_segments, avg_mask, input_positions)

                loss = criterion(outputs.float(), targets.float())

                test_loss += loss.item()
                loss_count += 1

                progress.update(batch+1, loss="%.4f" % loss.item())

        # mlflow.log_metric("test_loss", test_loss/loss_count)

        return {'loss': test_loss/loss_count}

class DistilBERT(BERT):
    """A simple DistilBERT system"""
    def load_tokenizer(self):
        return DistilBertTokenizer.from_pretrained("distilbert-base-uncased", cache_dir="huggingface_models")

    def load_bert_model(self):
        return {'hidden_size': 768}, DistilBertModel.from_pretrained("./distilbert-base-uncased", cache_dir="huggingface_models")

    def name(self):
        if self.positions:
            str_positions = "+pos"
        else:
            str_positions = ""
        if self.hidden_layer_size == 0:
            str_hidden = ""
        else:
            str_hidden = "-relu(%i)" % self.hidden_layer_size
        if self.finetune:
            str_finetune = "(finetuned)"
        else:
            str_finetune = "(frozen)"
        return "DistilBERT%s%s%s" % (str_finetune, str_positions, str_hidden)


if __name__ == "__main__":
    #import doctest
    import codecs
    #doctest.testmod()

    #sys.exit()

    #nnc = BERT(hidden_layer=50, build_model=False, positions=True)
    #nnc.fit(None, None, None, restore_model=True, verbose=0, savepath="task8b_nnc_model_1024")

    #question = "What is the best treatment for migraines?"
    #text = [
    #    "Exome results are reported for two patients with connective tissue dysplasia, one refining a clinical diagnosis of Ehlers-Danlos to Marfan syndrome, the other suggesting arthrogryposis derived from maternofetal Stickler syndrome.",
    #    "Patient 1 had mutations in transthyretin (TTR), fibrillin (FBN1), and a calcium channel (CACNA1A) gene suggesting diagnoses of transthyretin amyloidosis, Marfan syndrome, and familial hemiplegic migraines, respectively.",
    #    "Patient 2 presented with arthrogryposis that was correlated with his mother's habitus and arthritis once COL2A1 mutations suggestive of Stickler syndrome were defined.",
    #    "Although DNA results often defy prediction by the best of clinicians, these patients illustrate needs for ongoing clinical scholarship (e.g., to delineate guidelines for management of mutations like that for hyperekplexia in Patient 2) and for interpretation of polygenic change that is optimized by clinical genetic/syndromology experience (e.g., suggesting acetazolamide therapy for Patient 1 and explaining arthrogryposis in Patient 2)."
    #    ]
    #scores = summarise(question, text, nnc)
    #print(scores)

    #sys.exit()

    import csv

    def rouge_to_labels(rougeFile, labels, labelsthreshold, metric=["SU4"]):
        """Convert ROUGE values into classification labels
        >>> labels = rouge_to_labels("rouge_6b.csv", "topn", 3)
        Setting top 3 classification labels
        >>> labels[(0, '55031181e9bde69634000014', 9)]
        0.0
        >>> labels[(0, '55031181e9bde69634000014', 20)]
        1.0
        >>> labels[(0, '55031181e9bde69634000014', 3)]
        1.0
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
                result[qid_items[-1][0]] = 1.0
        else:
            print("Setting top", labelsthreshold, "classification labels")
            qids = set(k[0] for k in rouge)
            for qid in qids:
                qid_items = [(key, value) for key, value in rouge.items() if key[0] == qid]
                qid_items.sort(key = lambda k: k[1])
                for k, v in qid_items[-labelsthreshold:]:
                    result[k] = 1.0
                for k, v in qid_items[:-labelsthreshold]:
                    result[k] = 0.0
        return result

    nnc = DistilBERT(hidden_layer_size=50, positions=True, finetune_model=True)   

    print(nnc.name(), "created")



    labels_dict = rouge_to_labels('rouge_10b.csv', "topn", 5)
    sentences = []
    labels = []
    s_ids = []
    with open('rouge_10b.csv', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentences.append(nnc.cleantext(row['sentence text']))
            labels.append([labels_dict[(int(row['qid']), row['pubmedid'], int(row['sentid']))]])
            s_ids.append([int(row['sentid'])])
    print("Data has %i items" % len(sentences))
    #print(sentences[:3])
    #print(labels[:3])

    print("Training %s" % nnc.name())
    loss = nnc.fit(sentences[100:500], 
                   sentences[100:500], 
                   np.array(labels[100:500]),
                   X_positions=np.array(s_ids[100:500]),
                   verbose=2,
                   validation_data=(sentences[:100], sentences[:100], labels[:100], s_ids[:100]),
                   nb_epoch=3)
    print()
    print("Training loss of each epoch: %s" % (str(loss['loss'])))
    print("Validation loss of each epoch: %s" % (str(loss['val_loss'])))
    testloss = nnc.test(sentences[:100], 
                        sentences[:100], 
                        np.array(labels[:100]),
                        X_positions=np.array(s_ids[:100]))
    print("Test loss: %s" % (testloss['loss']))
    
