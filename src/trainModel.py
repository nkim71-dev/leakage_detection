import os, pickle, time, shutil
import numpy as np
from tensorflow import keras
from datetime import datetime
from tqdm import tqdm, trange
from sklearn.metrics import classification_report
from utils import makeDir, setGpuUsage
from model import Predictor_MLP
import argparse

parser = argparse.ArgumentParser(description='Parameters for model training')
parser.add_argument('--model-name', type=str, required=False, default='model_proposed_mlp', help='모델 이름')
parser.add_argument('--epochs', type=int, required=False, default=1000, help='최대 epochs')
parser.add_argument('--batch-size', type=int, required=False, default=256, help='배치 사이즈')
parser.add_argument('--repetition', type=int, required=False, default=10, help='앙상블에 사용할 모델 갯 수')
parser.add_argument('--gpu-memory', type=int, required=False, default=4096, help='GPU 사용량')


if __name__ == "__main__":
    dt_string_init = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    args = parser.parse_args()
    setGpuUsage(args.gpu_memory)
    makeDir('./models')

    with open('./data/processedData.pickle', 'rb') as handle:
        data_dict = pickle.load(handle)
    xDataTrain = data_dict['xDataTrain']
    yDataTrain = data_dict['yDataTrain']
    xDataValid = data_dict['xDataValid']
    yDataValid = data_dict['yDataValid']
    xDataTest = data_dict['xDataTest']
    yDataTest = data_dict['yDataTest']
    
    curr_dir = os.getcwd()
    dest_dir = f"models/{args.model_name}_{dt_string_init}"
    makeDir(dest_dir)        
    for i in range(args.repetition):
        
        predictor = Predictor_MLP(xDataTrain.shape[1])    
        predictor.compile(optimizer=keras.optimizers.Adam(1e-4, clipnorm=1.0))

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_LEAKAGE_loss', factor=0.5, patience=10, min_lr=1e-6)
        earlyStop = keras.callbacks.EarlyStopping(monitor="val_LEAKAGE_loss", patience=15, 
                                                restore_best_weights=True, start_from_epoch=30)
        history = predictor.fit(x=xDataTrain, y=yDataTrain,  
                                validation_data = (xDataValid, yDataValid),
                                epochs=args.epochs, batch_size=args.batch_size,
                                callbacks=[reduce_lr, earlyStop], verbose=1)
        
        dt_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        predictor.save_weights(os.path.join(f'{args.model_name}_{dt_string}','checkpoint')) 
        dir_to_move = os.path.join(curr_dir, f'{args.model_name}_{dt_string}')
        shutil.move(dir_to_move, dest_dir)

        preds, attn, probs = predictor.predict(xDataTest)
        print(f"{i+1}/{args.repetition}:")
        print(classification_report((yDataTest[:,0]>=0.5).astype('int'), np.argmax(probs, axis=-1).astype('int'), digits=4)) 
        print(np.mean(np.sum(-1 * np.array([1-yDataTest[:,0], yDataTest[:,0]]).T * np.log(probs),axis=-1)))

        