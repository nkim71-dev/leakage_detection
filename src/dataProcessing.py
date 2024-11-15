import numpy as np
import pandas as pd
import os, json, pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime
import argparse
from utils import timeseriesBatch

        
def loadProcessFile(filepath, scalers, xCols, yCols):
    df = pd.read_csv(filepath) if filepath.split('.')[-1]=='csv' else pd.read_parquet(filepath)
    df[xCols] = scalers['X'].transform(df[xCols])
    df[yCols] = scalers['Y'].transform(df[yCols])+0
    return df

parser = argparse.ArgumentParser(description='aggregate data (month)')
parser.add_argument('--chamber', type=int, default=1)     
args = parser.parse_args()
if __name__ == "__main__":
    dt_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    datapath = './labeledData'
    destPath = './data'

    filename = f'chamber{args.chamber}_train.parquet'
    df = pd.read_parquet(os.path.normpath(os.path.join(datapath, filename)))
    BOV = [col for col in df.columns if 'BOV' in col.upper()][0]
    IGV = [col for col in df.columns if 'IGV' in col.upper()][0]

    with open(os.path.normpath(os.path.join(destPath, 'columns.json')), 'r') as openfile:
        columnDict = json.load(openfile)
    yCols = columnDict['yColumns']
    xCols = columnDict['xColumns']
    
    scalers = dict()
    scaler = MinMaxScaler()
    scalers['X'] = scaler.fit(df[xCols])
    scaler = StandardScaler()
    scalers['Y'] = scaler.fit(df[yCols])

    with open(f'{destPath}/scalers.pickle', 'wb') as handle:
        pickle.dump(scalers, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    normalized_df = df.copy()
    normalized_df[xCols] = scalers['X'].transform(df[xCols])
    normalized_df[yCols] = scalers['Y'].transform(df[yCols])+0
    
    batchGen = timeseriesBatch(normalized_df)
    xDataTrain, yDataTrain = batchGen.generateTimeseries(xColumns=xCols, yColumns=['LEAKAGE']+yCols)
    xDataTrain = np.transpose(xDataTrain, (0, 2, 1))
    
    valid_file = f'chamber{args.chamber}_valid.parquet'
    filepath = os.path.normpath(os.path.join(datapath, valid_file))
    valid_df = loadProcessFile(filepath=filepath, scalers=scalers, xCols=xCols, yCols=yCols)
    validBatchGen = timeseriesBatch(valid_df)
    xDataValid, yDataValid = validBatchGen.generateTimeseries(xColumns=xCols, yColumns=['LEAKAGE']+yCols)
    xDataValid = np.transpose(xDataValid, (0, 2, 1))
    
    test_file = f'chamber{args.chamber}_test.parquet'
    filepath = os.path.normpath(os.path.join(datapath, test_file))
    test_df = loadProcessFile(filepath=filepath, scalers=scalers, xCols=xCols, yCols=yCols)
    testBatchGen = timeseriesBatch(test_df)
    xDataTest, yDataTest = testBatchGen.generateTimeseries(xColumns=xCols, yColumns=['LEAKAGE']+yCols)
    xDataTest = np.transpose(xDataTest, (0, 2, 1))

    df_merge = pd.concat([df, valid_df, test_df], axis=0).reset_index(drop=True)
    yData = np.concatenate([yDataTrain, yDataValid, yDataTest], axis=0)
    
    data_dict = {'xDataTrain': xDataTrain,
                 'yDataTrain': yDataTrain,
                'xDataValid': xDataValid,
                'yDataValid': yDataValid,
                'xDataTest': xDataTest,
                'yDataTest': yDataTest}
    
    with open(f'{destPath}/processedData.pickle', 'wb') as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

