from django.shortcuts import render
from django.http import HttpRequest

from .predict import predict
from .tf_predict import tf_predict

# Create your views here.

def index(request: HttpRequest):
    result = {}
    if request.method == 'POST':
        text = request.POST.get('text')
        model = request.POST.get('model')

        if model == 'pytorch':
            output, pred = predict(text)
        else:
            output, pred = tf_predict(text)

        result['output'] = output
        result['pred'] = pred
        result['text'] = text
        result['model'] = model
        result['text_length'] = len(text.split())

    return render(request, 'index.html', result)