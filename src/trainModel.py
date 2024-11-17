import os, pickle, shutil
import numpy as np
from tensorflow import keras
from datetime import datetime
from tqdm import tqdm, trange
from sklearn.metrics import classification_report
from utils import makeDir, setGpuUsage
from model import Predictor_MLP
import argparse

# 코드 입력 인자
parser = argparse.ArgumentParser(description='Parameters for model training')
parser.add_argument('--model-name', type=str, required=False, default='model_proposed_mlp', help='모델 이름')
parser.add_argument('--epochs', type=int, required=False, default=1000, help='최대 epochs')
parser.add_argument('--batch-size', type=int, required=False, default=256, help='배치 사이즈')
parser.add_argument('--repetition', type=int, required=False, default=10, help='앙상블에 사용할 모델 갯 수')
parser.add_argument('--gpu-memory', type=int, required=False, default=4096, help='GPU 사용량')


if __name__ == "__main__":

    # 코드 실행 준비
    dt_string_init = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") # 학습 코드 시작 일시
    args = parser.parse_args() # 학습에 필요한 argment 파싱
    setGpuUsage(args.gpu_memory) # 학습에 사용할 GPU메모리 설정
    makeDir('./models') # 학습 모델 저장 디렉토리 생성

    # 전처리 데이터 로드
    with open('./data/processedData.pickle', 'rb') as handle:
        data_dict = pickle.load(handle)
    xDataTrain = data_dict['xDataTrain']
    yDataTrain = data_dict['yDataTrain']
    xDataValid = data_dict['xDataValid']
    yDataValid = data_dict['yDataValid']
    xDataTest = data_dict['xDataTest']
    yDataTest = data_dict['yDataTest']
    
    # 현재 디렉토리 및 모델 저장 디렉토리 설정
    curr_dir = os.getcwd()
    dest_dir = f"models/{args.model_name}_{dt_string_init}"
    makeDir(dest_dir)        

    # 학습 수행
    for i in range(args.repetition):
        
        # 모델 선언 및 컴파일
        predictor = Predictor_MLP(xDataTrain.shape[1])    
        predictor.compile(optimizer=keras.optimizers.Adam(1e-4, clipnorm=1.0))

        # 학습을 위한 하이퍼파라미터 설정 (learning rate scheduler, early stopping)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_LEAKAGE_loss', factor=0.5, patience=10, min_lr=1e-6)
        earlyStop = keras.callbacks.EarlyStopping(monitor="val_LEAKAGE_loss", patience=15, 
                                                restore_best_weights=True, start_from_epoch=30)
        
        # 학습
        history = predictor.fit(x=xDataTrain, y=yDataTrain,  
                                validation_data = (xDataValid, yDataValid),
                                epochs=args.epochs, batch_size=args.batch_size,
                                callbacks=[reduce_lr, earlyStop], verbose=1)
        
        # 학습 모델 저장
        dt_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") # 모델 학습 완료 일시
        predictor.save_weights(os.path.join(f'{args.model_name}_{dt_string}','checkpoint')) # 모델 저장
        dir_to_move = os.path.join(curr_dir, f'{args.model_name}_{dt_string}') # 모델 저장 디렉토리 설정 
        shutil.move(dir_to_move, dest_dir) # destination 디렉토리로 모델 이동 (윈도우 기반 시스템에서 학습 진행 시 필요)

        # test data 기준 성능 확인
        proba, _ = predictor.predict(xDataTest)
        print(f"{i+1}/{args.repetition}:")
        print(classification_report((yDataTest[:,0]>=0.5).astype('int'), np.argmax(proba, axis=-1).astype('int'), digits=4)) 
        print(np.mean(np.sum(-1 * np.array([1-yDataTest[:,0], yDataTest[:,0]]).T * np.log(proba),axis=-1)))

        