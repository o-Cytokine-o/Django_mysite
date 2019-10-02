from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
import polls.janken_AI as jAI
import json
import numpy as np

# Create your views here.
def index(request):
    ram = [1,2,3]
    context = {
        'ram1':ram
    }
    return render(request, 'polls/index.html', context)

def detail(request, question_id):
    return HttpResponse("You're looking at question %s." % question_id)

def results(request, question_id):
    response = "You're looking at the results of question %s."
    return HttpResponse(response % question_id)

def vote(request, question_id):
    return HttpResponse("You're voting on question %s." % question_id)

#じゃんけん
def janken_main(request):
    return render(request, 'polls/janken_main.html')

def janken(request):
    
    return render(request, 'polls/janken.html')

def janken_choiced(request,choice_num):
    jankenAI = jAI.janken()



    return render(request, 'polls/janken.html')