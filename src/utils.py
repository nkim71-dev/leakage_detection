import os
import numpy as np
from datetime import datetime
import tensorflow as tf

def makeDir(path):
    if os.path.exists(path):
        return 'Already existing'
    os.mkdir(path)
    return 'Path created'

def setGpuUsage(memory=2048):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory)]
            )
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)


class timeseriesBatch():
    def __init__(self,df):
        self.df = df.reset_index(drop=True)
        self.df_list = self.removeTimeDiscontinuity()

    def generateTimeseries(self, xColumns, yColumns, infoColumns=None, window=5, shift=1):
        X, Y, info = [], [], []
        for df in self.df_list:
            if len(df)-window-shift<1:
                continue
            for row in range(len(df)-window-shift+1):
                endRow = row+window-1
                X.append(np.array(df.loc[row:endRow, xColumns], dtype='float32'))
                Y.append(np.array(df.loc[endRow+shift, yColumns], dtype='float32'))
                if infoColumns != None:
                    info.append(np.array(df.loc[endRow+shift, infoColumns]))
        X = np.stack(X, axis=0)
        Y = np.stack(Y, axis=0)
        if infoColumns != None:
            info = np.stack(info, axis=0)
            return X, Y, info
        return X, Y

    def removeTimeDiscontinuity(self):
        self.df["TIMESTAMP"] = self.df["TIME"].apply(lambda x: int((x-datetime(2023,4,1)).total_seconds()/60)) 
        skipVal = self.df.TIMESTAMP.diff() > 1
        if len(self.df.index[skipVal])==0:
            return [self.df]
        df_list = []
        row = 0
        for idx in self.df.index[skipVal]:
            df_list.append(self.df.iloc[row:idx].reset_index(drop=True))
            row = idx
        df_list.append(self.df.iloc[row:].reset_index(drop=True))
        return df_list
        