from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
import pandas as pd
from sklearn import preprocessing

def make_data(name, dfa):

    (rate, sig) = wav.read(name)

    mfcc_feat =mfcc(sig, rate, winlen=0.25, winstep=0.0125,
        numcep=13, nfilt=26, nfft=12500,
         highfreq= 22050, preemph=0.97)

    res_shape = np.shape(mfcc_feat)

    df = pd.DataFrame(mfcc_feat)

    if name.find("woman") != -1:

        df['SEX'] = pd.Series([0] * res_shape[0], index=df.index)
            
    elif name.find("man") != -1:
            
        df['SEX'] = pd.Series([1] * res_shape[0], index=df.index)

    else:
        print("Wrong name of the recording.")
        exit(1)
    dfa = dfa.append(df)

    return dfa


# Создаем пустой датафрейм, который и будем передавать функциям
df_all = pd.DataFrame()

i = 1
while i < 6:
    df_all = make_data("./trening/man{}.wav".format(i), df_all)
    df_all = make_data("./trening/woman{}.wav".format(i), df_all)
    i += 1

# переводим полученный датафрейм в формат cv
df_all.to_csv('filedata.csv')


# нормируем данные файла filedata.csv
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(df_all)
df_normalized = pd.DataFrame(np_scaled)
df_normalized.to_csv('filedata_norm.csv')


