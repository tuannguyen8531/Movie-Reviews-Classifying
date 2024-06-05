from django.shortcuts import render
from django.http import HttpRequest

from .predict import predict
from .binary_predict import predict_v2

# Create your views here.

def index(request: HttpRequest):
    result = {}
    if request.method == 'POST':
        text = request.POST.get('text')
        model = request.POST.get('model')
        if model == 'multiclass':
            output, pred = predict(text)
            result['output'] = output
            result['pred'] = pred
        else:
            output, pred = predict_v2(text)
            result['neg'] = format(100 - float(output), ".2f")
            result['pos'] = output
            result['pred'] = pred
        result['text'] = text
        result['model'] = model
        result['text_length'] = len(text.split())

    return render(request, 'index.html', result)