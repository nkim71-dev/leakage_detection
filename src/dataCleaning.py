import pandas as pd
import os
import csv
import argparse
from datetime import datetime
from utils import makeDir
import warnings
warnings.filterwarnings(action='ignore')

parser = argparse.ArgumentParser(description='aggregate data (month)')
parser.add_argument('--month', type=str, help='e.g., 202303')     
args = parser.parse_args()
if __name__ == "__main__":
    data_time = args.month
    datapath = './rawData'
    destPath = './cleanedData'
    filelist = [f for f in os.listdir(datapath) if (data_time in f) and ('merged_' not in f)]
    if len(filelist)==0:
        raise('file not exist')
    
    df_list = []
    for j_, filename in enumerate(filelist):
        value_list = []
        with open(os.path.join(datapath, filename)) as f:
            reader = csv.reader(f, delimiter="\t")
            for i_, line in enumerate(reader):
                if i_ == 0:
                    columns = line[0].split(',')
                    columns = [col.replace('.', '_') for col in columns]
                    continue
                value_list.append(line[0].split(','))
        df_list.append(pd.DataFrame(value_list, columns = columns))


    processed_columns = [col.replace("DUSAN_","").replace("통신_","").replace("캐스코드","Cascode").replace("시각","TIME") for col in columns]
    df_raw = pd.concat(df_list, axis=0)
    df_raw = df_raw.rename(columns={prev: new for prev, new in zip(columns, processed_columns)}).copy()

    columns_list = []
    df_chamber_list = []
    for i in range(2):
        columns_list.append([col for col in processed_columns if f"제{i+1}공압실" in col])
        df_tmp = df_raw[columns_list[i]]
        df_tmp["CHAMBER"] = i+1
        processed_cols = [col.replace(f"제{i+1}공압실_","") for col in columns_list[i]]
        df_tmp = df_tmp.rename(columns={prev: new for prev, new in zip(columns_list[i], processed_cols)})
        df_chamber_list.append(df_tmp)
    info_cols = [col for col in processed_columns if not sum([col in cols for cols in columns_list])]

    for i_ in range(len(df_chamber_list)):
        df_chamber_list[i_] = pd.concat([df_raw[info_cols], df_chamber_list[i_]], axis=1)

    df = pd.merge(df_chamber_list[0], df_chamber_list[1],"outer").dropna(axis=1).reset_index(drop=True)
    for col in df.columns:
        if col == "TIME":
            df["TIME"] = pd.to_datetime(df["TIME"])
        else:
            df[col] = pd.to_numeric(df[col])
    df = df.dropna(axis=0).copy()
    df_columns = df.columns.tolist()

    if "TIME" in df_columns:
        df_columns.remove('TIME')
    if "CHAMBER" in df_columns:
        df_columns.remove('CHAMBER')
    df["TIMESTAMP"] = df["TIME"].apply(lambda x: int((x-datetime(2023,4,1)).total_seconds()/60)) 
    df = df[["TIME","TIMESTAMP","CHAMBER"]+df_columns].sort_values(by=["CHAMBER","TIMESTAMP"]).reset_index(drop=True)

    BOV = [col for col in df_columns if 'BOV' in col.upper()][0]
    IGV = [col for col in df_columns if 'IGV' in col.upper()][0]

    makeDir(destPath)
    df.reset_index(drop=True).to_csv(f'{destPath}/cleaned_{data_time}.csv', index=False)