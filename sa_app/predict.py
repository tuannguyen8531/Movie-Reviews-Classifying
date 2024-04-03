import re
import nltk
import spacy
import spacy.cli
import torch
import pickle
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
if not spacy.util.is_package("en_core_web_sm"):
    spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    words = word_tokenize(text)

    filtered_words = [word for word in words if not word in stop_words and len(word) > 3]
    processed_text = ''
    for word in filtered_words:
        word = lemmatizer.lemmatize(word)
        processed_text = processed_text  + ' ' + str(word)

    doc = nlp(processed_text)
    processed_text = ''
    for token in doc:
        processed_text = processed_text  + ' ' + str(token.lemma_)

    return re.sub(r'\s+', ' ', processed_text)

def get_vocab():
    try:
        with open('sa_app/static/models/tokenizer.pkl', 'rb') as f:
            vocab = pickle.load(f)
    except FileNotFoundError as e:
        raise Exception(e)
    
    return vocab

def convert_pharse_to_int(text, vocab):
    text = clean_text(text)
    words = word_tokenize(text)
    text_ints = []
    text_ints.append([vocab[word] for word in words])

    return text_ints

def pad_sequences(phrase_to_int, seq_length):
    pad_sequences = np.zeros((len(phrase_to_int), seq_length), dtype=int)
    for idx, row in enumerate(phrase_to_int):
        pad_sequences[idx, :len(row)] = np.array(row)[:seq_length]
        
    return pad_sequences

vocab = get_vocab()
n_vocab = len(vocab) + 1
model = SentimentRNN(n_vocab, constants.N_OUTPUT, constants.N_EMBED, constants.N_HIDDEN, constants.N_LAYERS, constants.DEVICE)
model.load_state_dict(torch.load('sa_app/static/models/model_010424_040333.pth', map_location=torch.device(constants.DEVICE)))
model.to(constants.DEVICE)
model.eval()

def predict(sentence):
    text_ints = convert_pharse_to_int(sentence, vocab)
    padded_text = pad_sequences(text_ints, constants.SEQ_LENGTH)

    model.eval()

    text_tensor = torch.from_numpy(padded_text)
    batch_size = text_tensor.size(0)
    h = model.init_hidden(batch_size)
    text_tensor = text_tensor.to(constants.DEVICE)

    output, h = model(text_tensor, h)
    pred = torch.round(output.squeeze())

    return output.item(), pred.item()