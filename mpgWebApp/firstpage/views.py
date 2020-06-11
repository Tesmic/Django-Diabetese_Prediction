from django.shortcuts import render
from django.http import HttpResponse
import numpy as np
import pandas as pd
# Create your views here.
from sklearn.externals import joblib

classifier=joblib.load('./models/RFModelforMPG.pkl')


def index(request):
    temp={}
    temp['ageVal']=50
    temp['bs_fastVal']=6.8
    temp['bs_ppVal']=8.8
    temp['plasma_rVal']=11.2
    temp['plasma_fVal']=7.2
    temp['hbA1cVal']=62
    context ={'temp':temp}
    return render(request,'index.html',context)
 #   return HttpResponse({'a':1})

def predictMPG(request):
    context ={'temp':'  '}
    print (request)
    if request.method == 'POST':
        temp={}
        temp['ageVal']=request.POST.get('ageVal')
        temp['bs_fastVal']=request.POST.get('bs_fastVal')
        temp['bs_ppVal']=request.POST.get('bs_ppVal')
        temp['plasma_rVal']=request.POST.get('plasma_rVal')
        temp['plasma_fVal']=request.POST.get('plasma_fVal')
        temp['hbA1cVal']=request.POST.get('hbA1cVal')
#   testDtaa=pd.DataFrame({'x':temp2}).transpose()
#  scoreval=reloadModel.predict(testDtaa)[0]
#   context={'scoreval':scoreval,'temp':temp}
        #Datapreprocessing Convert the values to float
        age = float(temp['ageVal'])
        bs_fast = float(temp['bs_fastVal'])
        bs_pp = float(temp['bs_ppVal'])
        plasma_r = float(temp['plasma_rVal'])
        plasma_f = float(temp['plasma_fVal'])
        hbA1c = float(temp['hbA1cVal'])
        result = [age,bs_fast,bs_pp,plasma_r,plasma_f,hbA1c]
        #Passing data to model & loading the model from disks
        #model_path = 'ml_model/model.pkl'
        #classifier = pickle.load(open(model_path, 'rb'))
        prediction = classifier.predict([result])[0]
        conf_score =  np.max(classifier.predict_proba([result]))*100
        context ={'a':'Diabetes Prediction :' + prediction ,'temp':temp}
    return render(request,'index.html',context)