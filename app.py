from flask import Flask,request,jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import torch.nn as nn
import torch


from sklearn.externals import joblib

app = Flask(__name__)

cors = CORS(app) 

class SoftmaxRegressionModel(nn.Module):
    
    def __init__(self):
        super().__init__()

        
        self.fc1 = nn.Linear(52,13)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(13,3)

    def forward(self,x):

        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        return out



def predict(x,y):
    
    single_x = np.array([x,y])
    
    labelencoder_team = joblib.load('label.pkl')
    single_x[0] = labelencoder_team.transform(single_x[0].reshape(-1))[0]
    single_x[1] = labelencoder_team.transform(single_x[1].reshape(-1))[0]
          
    onehotencoder = joblib.load('onehot.pkl')
    single_x = onehotencoder.transform(single_x.reshape(1,-1)).toarray()
    
    single_x =pd.DataFrame(single_x )
    
    single_x.drop([0,51],axis=1,inplace= True)
    
    single_x[51]=0.8
    single_x[52]=0.4
    
    single_x=single_x.values
    
    scale = joblib.load('scale.pkl')
    
    single_x = scale.transform(single_x)
    
    
    model = joblib.load('model.pkl')
    
    data_input = torch.FloatTensor(single_x)
    
    outputs = model(data_input)
    m = nn.Softmax()
    
    final_output={'a':m(outputs)[0][0].item(),'b':m(outputs)[0][1].item(),'c':m(outputs)[0][2].item() }
    
   
    return final_output
    


@app.route('/parameters',methods=['POST'])

def get_parameters():
    
    usr_data=request.get_json()
    result=predict(usr_data['home'],usr_data['away'])

    return jsonify(result)

@app.route('/')

def homepage():
    

    return "Welcome to our prediction"








