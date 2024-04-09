import re
import nltk
import spacy
import torch
import pickle
import glob
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from .models import SentimentRNN
from . import constants

def download_nltk_resource(package, resource_name):
    try:
        nltk.data.find(resource_name)
    except LookupError:
        nltk.download(package)

download_nltk_resource('punkt', 'tokenizers/punkt')
download_nltk_resource('wordnet', 'corpora/wordnet')
download_nltk_resource('stopwords', 'corpora/stopwords')

lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    words = word_tokenize(text)

    filtered_words = [word for word in words if not word in stop_words and len(word) > 3]
    lemmas = []
    for word in filtered_words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    sentence = ' '.join(lemmas)
    doc = nlp(sentence)
    lemmas = [token.lemma_ for token in doc]

    return lemmas

def get_vocab():
    try:
        with open('sa_app/static/models/tokenizer.pkl', 'rb') as f:
            vocab = pickle.load(f)
    except FileNotFoundError as e:
        raise Exception(e)
    
    return vocab

def phrase_to_ints(words):
    ints = [vocab.get(word, vocab['<UNK>']) for word in words]
    return ints

def pad_sequences(phrase_to_int, seq_length):
    if not isinstance(phrase_to_int[0], list):
        phrase_to_int = [phrase_to_int]
    pad_sequences = np.zeros((len(phrase_to_int), seq_length), dtype=int)
    for idx, row in enumerate(phrase_to_int):
        pad_sequences[idx, :len(row)] = np.array(row)[:seq_length]
        
    return pad_sequences

vocab = get_vocab()
n_vocab = len(vocab) + 1

model_dir = 'sa_app/static/models/'
model_files =  glob.glob(model_dir + 'model_*.pth')
try:
    model_fname = model_files[0]
    model = SentimentRNN(n_vocab, constants.N_OUTPUT, constants.N_EMBED, constants.N_HIDDEN, constants.N_LAYERS, constants.DEVICE)
    model.load_state_dict(torch.load(model_fname, map_location=torch.device(constants.DEVICE)))
except IndexError:
    raise Exception('File not found')
except Exception as e:
    raise Exception(e)

model.to(constants.DEVICE)
model.eval()

def predict(sentence):
    words = clean_text(sentence)
    text_ints = phrase_to_ints(words)
    padded_text = pad_sequences(text_ints, constants.SEQ_LENGTH)
    
    text_tensor = torch.from_numpy(padded_text)
    text_tensor = text_tensor.to(constants.DEVICE)
    
    model.eval()
    
    batch_size = text_tensor.size(0)
    h = model.init_hidden(batch_size)
    
    output, h = model(text_tensor, h)
    pred = torch.round(output.squeeze())
    
    return output.item(), pred.item()