from django.shortcuts import render
from django.http import HttpRequest

from .predict import predict

# Create your views here.

def index(request: HttpRequest):
    result = {}
    if request.method == 'POST':
        text = request.POST.get('text')
        output, pred = predict(text)
        result['output'] = output
        result['pred'] = pred
        result['text'] = text
        result['text_length'] = len(text.split())

    return render(request, 'index.html', result)