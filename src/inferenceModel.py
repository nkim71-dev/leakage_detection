import numpy as np
import pandas as pd
import os, pickle
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import seaborn as sns

from utils import makeDir, setGpuUsage
from model import Predictor_MLP
import argparse
      

parser = argparse.ArgumentParser(description='Parameters for model inference')
parser.add_argument('--model-name', type=str, required=False, default=None, help='모델 이름')
parser.add_argument('--gpu-memory', type=int, required=False, default=4096, help='GPU 사용량')

if __name__ == "__main__":
    args = parser.parse_args()
    setGpuUsage(args.gpu_memory)
    makeDir('./figures')
    dt_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if args.model_name is None:
        model_list = os.listdir('./models')
        model_list.sort()
        model_name = model_list[-1]
    else:
        model_name = args.model_name

    print("Load preprocessed data")
    with open('./data/processedData.pickle', 'rb') as handle:
        data_dict = pickle.load(handle)    
    xDataTrain = data_dict['xDataTrain']
    yDataTrain = data_dict['yDataTrain']
    xDataValid = data_dict['xDataValid']
    yDataValid = data_dict['yDataValid']
    xDataTest = data_dict['xDataTest']
    yDataTest = data_dict['yDataTest']

    yTest = yDataTest[:,0]
    y_hard_true = (yTest>=0.5).astype('int')       
    predictor = Predictor_MLP(xDataTrain.shape[1])
    model_dir = os.path.join(os.getcwd(),'models',model_name)
    model_path = [dir for dir in os.listdir(model_dir)]

    test_prob_list = list()
    test_val_list = list()
    results_df = list()
    print(f"Inference models in '{model_name}'")
    for trained_model in tqdm(model_path):
        predictor.load_weights(os.path.join(model_dir,trained_model,'checkpoint')).expect_partial()
        testPreds, testAttn, testProbs = predictor.predict(xDataTest, verbose=0)
        test_prob_list.append(testProbs)
        test_val_list.append(np.concatenate(testPreds,axis=-1))
        
        y_hard_pred = (testProbs[:,1]>=0.5).astype('int')
        recall = (y_hard_true*y_hard_pred).sum()/(y_hard_true).sum()
        precision = (y_hard_true*y_hard_pred).sum()/(y_hard_pred).sum()
        accuracy = ((y_hard_true*y_hard_pred).sum() + ((1-y_hard_true)*(1-y_hard_pred)).sum())/len(y_hard_true)
        f1score = 2*(precision*recall)/(precision+recall)
        xEnt = log_loss(y_hard_true, testProbs[:,1])
        results_df.append({'model': trained_model, 'ACC': accuracy, 'F1': f1score, 'xEnt': xEnt})
    
    print(f"Ensemble models in '{model_name}'")
    y_prob_pred = np.mean(test_prob_list,axis=0)
    y_hard_pred = (y_prob_pred[:,1]>=0.5).astype('int')
    recall = (y_hard_true*y_hard_pred).sum()/(y_hard_true).sum()
    precision = (y_hard_true*y_hard_pred).sum()/(y_hard_pred).sum()
    accuracy = ((y_hard_true*y_hard_pred).sum() + ((1-y_hard_true)*(1-y_hard_pred)).sum())/len(y_hard_true)
    f1score = 2*(precision*recall)/(precision+recall)
    xEnt = log_loss(y_hard_true, y_prob_pred[:,1])
    print(f'ACC: {accuracy:04f} \nF1: {f1score:04f} \nxEnt: {xEnt:03f}\nConfusion Matrix:')
    print(classification_report(y_hard_true, y_hard_pred, digits=4))

    with open('./data/scalers.pickle', 'rb') as handle:
        scalers = pickle.load(handle)
    for idx in range(len(test_val_list)):
        preds_df = pd.DataFrame(scalers['Y'].inverse_transform(test_val_list[idx]), columns = ['BOV (Prediction)', 'IGV (Prediction)'])
        preds_df[['BOV', 'IGV']] = scalers['Y'].inverse_transform(yDataTest[:,1:3])
        preds_df['Predicted Class'] =  pd.DataFrame((test_prob_list[idx][:,1]>=0.5).astype('int')).map(lambda x: 'Leakage' if x==1 else 'Non-leakage') 
        preds_df['Actual Class'] =  pd.DataFrame(yDataTest[:,0]>=0.5).astype('int').map(lambda x: 'Leakage' if x==1 else 'Non-leakage') 
        
    preds_df = pd.DataFrame(scalers['Y'].inverse_transform(np.mean(test_val_list,axis=0)), columns = ['BOV (Prediction)', 'IGV (Prediction)'])
    preds_df[['BOV', 'IGV']] = scalers['Y'].inverse_transform(yDataTest[:,1:3])
    preds_df['Predicted Class'] =  pd.DataFrame((np.mean(test_prob_list,axis=0)[:,1]>=0.5).astype('int')).map(lambda x: 'Leakage' if x==1 else 'Non-leakage') 
    preds_df['Actual Class'] =  pd.DataFrame(yDataTest[:,0]>=0.5).astype('int').map(lambda x: 'Leakage' if x==1 else 'Non-leakage') 

    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots(1,1,figsize=(5,2))
    ax = sns.scatterplot(data=preds_df, x='BOV (Prediction)', y='IGV (Prediction)', hue='Actual Class', style='Predicted Class', alpha=0.8)
    ax.set_ylabel('IGV (Projected)')
    ax.set_xlabel('BOV (Projected)')
    ax.legend(ncol=2)
    ax.grid()
    fig.tight_layout()
    fig.savefig(f'./figures/IGV_BOV_projected_{model_name}.pdf')
    plt.clf()
    
    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots(1,1,figsize=(5,2))
    ax = sns.scatterplot(data=preds_df, x='BOV', y='IGV', hue='Actual Class', style='Predicted Class', alpha=0.8)
    ax.set_ylabel('IGV (Ground Truth)')
    ax.set_xlabel('BOV (Ground Truth)')
    ax.legend(ncol=2)
    ax.grid()
    fig.tight_layout()
    fig.savefig(f'./figures/IGV_BOV_original{model_name}.pdf')
    plt.clf()