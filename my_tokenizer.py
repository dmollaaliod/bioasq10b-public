import nltk
from nltk.corpus import stopwords

def my_sent_tokenize(text):
    try:
        sent_tokens = nltk.sent_tokenize(text)
    except:
        print('Removing leading "%s" since it causes error in sent_tokenize' % (text[0]))
        text = text[1:]
        sent_tokens = nltk.sent_tokenize(text)
    return sent_tokens

def my_tokenize(string):
    """Return the list of tokens.
    >>> my_tokenize("This is a sentence. This is another sentence.")
    ['sentence', 'another', 'sentence']
    """
    return [w.lower() 
            for s in nltk.sent_tokenize(string) 
            for w in nltk.word_tokenize(s)
            if w.lower() not in stopwords.words('english') and
               w not in [',','.',';','(',')','"',"'",'=',':','%','[',']']]
 
