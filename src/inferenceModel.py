import numpy as np
import pandas as pd
import os, pickle
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import log_loss, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from utils import makeDir, setGpuUsage
from model import Predictor_MLP
import argparse
      
# 코드 입력 인자
parser = argparse.ArgumentParser(description='Parameters for model inference')
parser.add_argument('--model-name', type=str, required=False, default=None, help='모델 이름')
parser.add_argument('--gpu-memory', type=int, required=False, default=4096, help='GPU 사용량')

if __name__ == "__main__":
    
    # 코드 실행 준비
    dt_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") # 성능 확인 코드 시작 일시
    args = parser.parse_args() # 추론에 필요한 argment 파싱
    setGpuUsage(args.gpu_memory) # 추론에 사용할 GPU메모리 설정
    makeDir('./figures') # 추론 결과 저장 디렉토리 생성

    # 전처리 데이터 로드
    print("Load preprocessed data")
    with open('./data/processedData.pickle', 'rb') as handle:
        data_dict = pickle.load(handle)    
    xDataTrain = data_dict['xDataTrain']
    yDataTrain = data_dict['yDataTrain']
    xDataValid = data_dict['xDataValid']
    yDataValid = data_dict['yDataValid']
    xDataTest = data_dict['xDataTest']
    yDataTest = data_dict['yDataTest']

    # Air leakage 라벨
    yTest = yDataTest[:,0]
    y_hard_true = (yTest>=0.5).astype('int')

    # 추론 모델 미지정 시 최신 버전의 모델 추론
    if args.model_name is None:
        model_list = os.listdir('./models')
        model_list.sort()
        model_name = model_list[-1]
    else:
        model_name = args.model_name
    model_dir = os.path.join(os.getcwd(),'models',model_name)
    model_path = [dir for dir in os.listdir(model_dir)]

    test_proba_list = list()
    test_bov_igv_list = list()
    results_df = list()
    print(f"Inference models in '{model_name}'")
    for trained_model in tqdm(model_path):
        # 추론
        predictor = Predictor_MLP(xDataTest.shape[1]) # 모델 선언
        predictor.load_weights(os.path.join(model_dir,trained_model,'checkpoint')).expect_partial() # 모델 가중치 로드
        predProba, projBovIgv = predictor.predict(xDataTest, verbose=0) # 추론
        test_proba_list.append(predProba) # leakage probability 결과 
        test_bov_igv_list.append(np.concatenate(projBovIgv,axis=-1)) # leakage probability 결과
        
        # 각 모델 별 추론 결과에 대한 성능 계산
        y_hard_pred = (predProba[:,1]>=0.5).astype('int')
        recall = (y_hard_true*y_hard_pred).sum()/(y_hard_true).sum()
        precision = (y_hard_true*y_hard_pred).sum()/(y_hard_pred).sum()
        accuracy = ((y_hard_true*y_hard_pred).sum() + ((1-y_hard_true)*(1-y_hard_pred)).sum())/len(y_hard_true)
        f1score = 2*(precision*recall)/(precision+recall)
        xEnt = log_loss(y_hard_true, predProba[:,1])
        results_df.append({'model': trained_model, 'ACC': accuracy, 'F1': f1score, 'xEnt': xEnt})
    
    # 모델들의 추론 결과 앙상블
    print(f"Ensemble models in '{model_name}'")
    y_prob_pred = np.mean(test_proba_list,axis=0)
    y_hard_pred = (y_prob_pred[:,1]>=0.5).astype('int')

    # 앙상블 결과에 대한 성능 계산
    recall = (y_hard_true*y_hard_pred).sum()/(y_hard_true).sum()
    precision = (y_hard_true*y_hard_pred).sum()/(y_hard_pred).sum()
    accuracy = ((y_hard_true*y_hard_pred).sum() + ((1-y_hard_true)*(1-y_hard_pred)).sum())/len(y_hard_true)
    f1score = 2*(precision*recall)/(precision+recall)
    xEnt = log_loss(y_hard_true, y_prob_pred[:,1])
    print(f'ACC: {accuracy:04f} \nF1: {f1score:04f} \nxEnt: {xEnt:03f}\nConfusion Matrix:')
    print(classification_report(y_hard_true, y_hard_pred, digits=4))

    # BOV, IGV projection 결과 시각화
    with open('./data/scalers.pickle', 'rb') as handle:
        scalers = pickle.load(handle)
    for idx in range(len(test_bov_igv_list)):
        preds_df = pd.DataFrame(scalers['Y'].inverse_transform(test_bov_igv_list[idx]), columns = ['BOV (Prediction)', 'IGV (Prediction)'])
        preds_df[['BOV', 'IGV']] = scalers['Y'].inverse_transform(yDataTest[:,1:3])
        preds_df['Predicted Class'] =  pd.DataFrame((test_proba_list[idx][:,1]>=0.5).astype('int')).map(lambda x: 'Leakage' if x==1 else 'Non-leakage') 
        preds_df['Actual Class'] =  pd.DataFrame(yDataTest[:,0]>=0.5).astype('int').map(lambda x: 'Leakage' if x==1 else 'Non-leakage') 
        
    preds_df = pd.DataFrame(scalers['Y'].inverse_transform(np.mean(test_bov_igv_list,axis=0)), columns = ['BOV (Prediction)', 'IGV (Prediction)'])
    preds_df[['BOV', 'IGV']] = scalers['Y'].inverse_transform(yDataTest[:,1:3])
    preds_df['Predicted Class'] =  pd.DataFrame((np.mean(test_proba_list,axis=0)[:,1]>=0.5).astype('int')).map(lambda x: 'Leakage' if x==1 else 'Non-leakage') 
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