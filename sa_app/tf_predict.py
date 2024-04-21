import numpy as np
import pickle
import glob

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

try:
    with open('sa_app/static/models/tensorflow/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
except FileNotFoundError as e:
        raise Exception(e)

model_dir = 'sa_app/static/models/tensorflow/'
model_files =  glob.glob(model_dir + 'model_*.h5')

try:
    model_fname = model_files[0]
    model = load_model(model_fname)
except IndexError:
    raise Exception('File not found')
except Exception as e:
    raise Exception(e)

def tf_predict(sentence):
    new_complaint = [sentence]
    seq = tokenizer.texts_to_sequences(new_complaint)
    padded = pad_sequences(seq, maxlen=300)
    output = model.predict(padded)
    pred = np.argmax(output)

    output = output.tolist()
    output = output[0]

    return output, pred