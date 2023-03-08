from django.shortcuts import render
from .utils.sentiment import sentiment_classification

# Create your views here.

def sentiment_view(request):
    output='TBD'
    
    if request.method =='POST':
        
        text_value=request.POST.get("text_value")


        output=sentiment_classification(text_value)

    context={'output_sentiment':output}

    return render(request,'cyberbullyapp/base.html',context)