import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import argparse
from utils import makeDir
import warnings
warnings.filterwarnings(action='ignore')




def find_interseciton(m1,m2,std1,std2, w1, w2):
    import numpy as np
    a = 1/(2*std1**2) - 1/(2*std2**2)
    b = m2/(std2**2) - m1/(std1**2)
    c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1*w1/w2)
    return np.roots([a,b,c])

def gaussianValue(x, mu, sig):
    return [np.exp(-1/2*((x-m)/s)**2)/(s*(2*np.pi)**0.5) for m, s in zip(mu, sig)]

def labelLeakage(df, mu, sig):
    g_value = gaussianValue(df['BOV']-df['IGV'], mu[[1,2]], sig[[1,2]])
    g_value = np.array(g_value)
    g_value /= g_value.sum(axis=0)
    df['LEAKAGE'] = g_value[-1].round(2)
    return df.copy()

parser = argparse.ArgumentParser(description='aggregate data (month)')
parser.add_argument('--chamber', type=int, default=1)     
args = parser.parse_args()
if __name__ == "__main__":
    datapath = './mergedData'
    destPath = './labeledData'

    df = pd.read_csv(f'{datapath}/merged_chamber{args.chamber}.csv')
    df['TIME'] = pd.to_datetime(df['TIME'])
    df = df.drop(index=df.query("BOV==100").query("IGV==0").index).copy()
    df = df.drop(index=df.query("BOV==0").query("IGV==100").index).copy()

    df_train_list = list()
    df_valid_list = list()
    df_test_list = list()
    for months in range(4,9):
        df_tmp = df.loc[(df['TIME'].map(lambda x:x.month)==months)].sort_values(by=['TIME'])
        valid_start = int(len(df_tmp)*0.6)
        test_start = int(len(df_tmp)*0.8)
        df_train_list.append(df_tmp.iloc[:valid_start])
        df_valid_list.append(df_tmp.iloc[valid_start:test_start])
        df_test_list.append(df_tmp.iloc[test_start:])
    df_train = pd.concat(df_train_list, axis=0).reset_index(drop=True)
    df_valid = pd.concat(df_valid_list,axis=0).reset_index(drop=True)
    df_test = pd.concat(df_test_list,axis=0).reset_index(drop=True)
    

    df_fit = df_train.copy()
    gmm = GaussianMixture(n_components=4, means_init=[[-110], [-30], [30], [110]],
                      random_state=1234)
    gmm.fit(np.array(df_fit['BOV']-df_fit['IGV']).reshape(-1,1))
    fig, ax = plt.subplots(1,1,figsize=(5,3))
    plt.hist(df_fit['BOV']-df_fit['IGV'],bins=20, density=True, alpha=0.3, label='Actual Data Distribution')
    gmm.converged_
    mu = gmm.means_.squeeze()
    sig = gmm.covariances_.squeeze()**0.5
    ws = gmm.weights_
    x = np.arange(-100,100,1)
    tmp = np.zeros(x.shape)
    flag = 1
    for m, s, w in zip(mu, sig, ws):
        if flag:
            plt.plot(x,norm.pdf(x, loc=m, scale=s)*w, linewidth=2, c='tab:green', linestyle=':', label='Gaussian Distributions')
            flag =0
        else:
            plt.plot(x,norm.pdf(x, loc=m, scale=s)*w, linewidth=2, c='tab:green', linestyle=':')
        tmp += norm.pdf(x, loc=m, scale=s)*w
    x_i = [m for m in find_interseciton(*mu[1:3], *sig[1:3], *ws[1:3])][-1] #x[find_peaks(-tmp)[0][-2]]

    plt.plot(x,tmp,c='tab:red', linewidth=2, linestyle='--',label='Gaussian Mixture Model')
    plt.plot([x_i,x_i],[0,max(tmp)], linewidth=2, c='tab:orange', label='Decision Boundary')
    plt.xlabel('Airflow Correction (BOV-IGV)')
    plt.ylabel('Probability Density')
    plt.legend()
    fig.tight_layout()

    makeDir('./figures')
    fig.savefig(f'./figures/leakage_labeling_with_gmm.pdf')
    # print([m for m in find_interseciton(*mu[1:3], *sig[1:3], *ws[1:3])])
   
    df_train = labelLeakage(df_train, mu, sig)
    df_valid = labelLeakage(df_valid, mu, sig)
    df_test = labelLeakage(df_test, mu, sig)

    makeDir(destPath)
    df_train.to_parquet(f'./labeledData/chamber{args.chamber}_train.parquet')
    df_valid.to_parquet(f'./labeledData/chamber{args.chamber}_valid.parquet')
    df_test.to_parquet(f'./labeledData/chamber{args.chamber}_test.parquet')
    df = pd.concat([df_train, df_valid, df_test],axis=0)
    df.to_parquet(f'./labeledData/chamber{args.chamber}_data.parquet')
