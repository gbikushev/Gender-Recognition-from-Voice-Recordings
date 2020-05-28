from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
import pandas as pd
from sklearn import preprocessing
import pickle
import speech_recognition as sr
import pyaudio


def make_data(name, dfa):
    (rate, sig) = wav.read(name)
    mfcc_feat =mfcc(sig, rate, winlen=0.2, winstep=0.01,
         nfilt=26, nfft=12512,lowfreq = 0,
         highfreq=22050, preemph=0)

    res_shape = np.shape(mfcc_feat)
    df = pd.DataFrame(mfcc_feat)
    dfa = dfa.append(df)
    return dfa

def voice_to_numpy(name):

    frame = pd.DataFrame()

    frame = make_data(name, frame)

    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(frame)

    frame_normalized = pd.DataFrame(np_scaled)

    data = frame_normalized.to_numpy()

    return data

key = int(input("1 - knn \n"
                "2 - rfc  \n"
                "Выберете класификатор:"))
m = int(input("\nВыберете способ проверки (1-записать голос / 2-использовать данные):"))

if m == 1:

    import speech_recognition as sr
    import pyaudio
    import scipy.io.wavfile as wav

    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        print(i, p.get_device_info_by_index(i)['name'])
    import wave

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 4
    WAVE_OUTPUT_FILENAME = "man.wav"
    p = pyaudio.PyAudio()
    i = int(input("\nВыберете свой микровон из списка (нажмите индекс):"))
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=i,
                    frames_per_buffer=CHUNK)
    print("\n* recording (говорите на любом языке)")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("* done recording \n")
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    data = voice_to_numpy(WAVE_OUTPUT_FILENAME)


if m == 2:
 name = input("\nВведите название :(язык/пол)\n"
              "Выбор языков :\n"
              "Azerbaijan\n"
                "English\n"
             "French\n"
                "Georgian\n" 
               "German\n"
                "Russian\n"
              "Spanish\n"
                "Vietnam\n"
              "Пример: English/man\n")

 data = voice_to_numpy("C:/Users/БикушевГлебДмитриеви/PycharmProjects/НИС/languages/{}.wav".format(name))


    # Загрузка сохраненной с помощью Pickle модели
if key == 1:
    with open("knn_best_model.sav", "rb") as fin:
        model = pickle.load(fin)

elif key == 2:
    with open("rfc_best_model.sav", "rb") as fin:
        model = pickle.load(fin)


    # Считаем среднее по столбцам предсказания поля
result = np.mean(model.predict(data))

if result < 0.5:

    print("Женщина")
    print(result)

elif result > 0.5:

    print("Мужчина")
    print(result)

else:

    print("Неопределенно.")
    print(result)