import pandas as pd
import numpy as np
import os
from utils import makeDir

if __name__ == "__main__":
    # 주어진 입력 데이터 디렉토리 데이터에서 내 chamber 별로 데이터 취합
    
    datapath = './cleanedData'
    destPath = './mergedData'
    
    df_list = [pd.read_csv(os.path.join(datapath, filename)) for filename in os.listdir(datapath)]
    df = pd.concat(df_list, axis=0)

    columns = df.columns.tolist()
    nonturbo_cols = [col for col in df.columns if 'TURBO' not in col]
    turbo_list = np.sort([col for col in set(sum([col.split('_') for col in columns if 'TURBO' in col],[])) if 'TURBO' in col])

    makeDir(destPath)
    for chamber in df['CHAMBER'].unique().tolist():
        df_chamber = df.query(f"CHAMBER=={chamber}").reset_index(drop=True)
        df_merged = None
        df_chamber_noTurbo = df_chamber.drop(columns=[col for col in df_chamber.columns if 'TURBO' in col])
        for turbo in turbo_list:
            turbo_cols = [col for col in columns if turbo in col]
            df_tmp = df_chamber[turbo_cols].rename(columns={prev: new for prev, new in zip(turbo_cols, [col.replace(f"{turbo}_", "") for col in turbo_cols])}).copy()
            df_tmp["TURBO"] = int(turbo[5:])
            if df_merged is None:
                df_merged = pd.concat([df_chamber_noTurbo,df_tmp],axis=1)
            else:
                df_merged = pd.concat([df_merged, pd.concat([df_chamber_noTurbo,df_tmp],axis=1)],axis=0).dropna(axis=1).reset_index(drop=True)
        
        df_merged = df_merged.replace('',np.nan).dropna(axis=0).reset_index(drop=True).copy()
        df_merged['TIME'] = pd.to_datetime(df_merged['TIME'])
        float_cols = [col for col in df_merged.columns if col not in ['TIME']]
        df_merged[float_cols] = df_merged[float_cols].astype('float')
        df_merged[['BOV', 'IGV', 'START_STOP']] = df_merged[['BOV', 'IGV', 'START_STOP']].astype('int')
        df_merged.query("START_STOP==1").to_csv(f'{destPath}/merged_chamber{chamber}.csv', index=False)
        # drop_cond = (df_merged[[col for col in df_merged.columns if col!='TIME']].agg(['std']).T<1e-1)
        # df_merged.drop(columns=drop_cond.loc[drop_cond['std']==True].T.columns).to_csv(f'{destPath}/merged_chamber{chamber}.csv', index=False)