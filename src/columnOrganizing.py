import pandas as pd
import os
import json
from utils import makeDir

if __name__ == "__main__":
    datapath = './labeledData'
    destPath = './data'
    makeDir(destPath)

    filelist = [f for f in os.listdir(datapath)]
    columns = None
    for filename in filelist:
        if columns == None:
            columns = set(pd.read_parquet(os.path.join(datapath, filename)).columns)
            continue
        columns = columns.intersection(set(pd.read_parquet(os.path.join(datapath, filename)).columns))
    df_columns = list(columns)
    
    infoCols = ["TIME", "TIMESTAMP", "CHAMBER"]
    BOV = [col for col in df_columns if 'BOV' in col.upper()][0]
    IGV = [col for col in df_columns if 'IGV' in col.upper()][0]
    yCols = [col for col in df_columns if ("IGV" in col) or ("BOV" in col)]
    xCols = [col for col in df_columns if (col not in yCols) and (col not in infoCols) and (col != 'LEAKAGE')]
    
    infoCols.sort()
    yCols.sort()
    xCols.sort()

    columnDict = dict()
    columnDict["infoColumns"] = infoCols
    columnDict["yColumns"] = yCols
    columnDict["xColumns"] = xCols
    columnDict["Leakage"] = ["LEAKAGE"]

    with open(f"{destPath}/columns.json", "w") as outfile:
        json.dump(columnDict, outfile)

    
    
    
