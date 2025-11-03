import librosa
import os
import numpy as np
import pandas as pd
import soundfile as sf
from noisereduce import reduce_noise


#1.Preprocessing
def preprocessing(input_path,output_path,target_sr = 16000):
    
    #1.loading the audio file
    y,sr = librosa.load(input_path,sr = None)


    #2.Resampling
    if sr != target_sr:
        
        y = librosa.resample(y,orig_sr = sr,target_sr = target_sr)
        sr = target_sr


    #3.Convert to mono (single channel)
    if len(y.shape) > 1:
        y = librosa.to_mono(y)


    #4.Trimming leading/trailing silence
    y, _ = librosa.effects.trim(y,top_db = 25)


    #5.Normalizing the amplitude
    y = y / np.max(np.abs(y))


    #6.Noise Reduction
    y = reduce_noise(y = y,sr = sr)
    

    #7.Saving the preprocessed file into a folder
    os.makedirs(os.path.dirname(output_path),exist_ok = True)
    sf.write(output_path,y,sr)
    return output_path


#2.Feature Extraction
def extract_features(file_path,target_sr=16000):
   
    #1.Loading the data
    y,sr = librosa.load(file_path,sr = None)

    #2.MFCC
    mfccs = librosa.feature.mfcc(y = y,sr = sr,n_mfcc = 40)
    mfccs_mean = np.mean(mfccs.T,axis = 0)

    #3.Chroma
    stft = np.abs(librosa.stft(y))
    chroma = librosa.feature.chroma_stft(S = stft,sr = sr)
    chroma_mean = np.mean(chroma.T,axis = 0)
    
    #4.Spectral Contrast
    contrast = librosa.feature.spectral_contrast(S = stft,sr = sr)
    contrast_mean = np.mean(contrast.T,axis = 0)

    #5.Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr.T,axis = 0)

    #6.Root Mean Square
    rms = librosa.feature.rms(y = y)
    rms_mean = np.mean(rms.T,axis = 0)

    #7.Mel Spectrogram
    mel = librosa.feature.melspectrogram(y = y,sr = sr)
    mel_mean = np.mean(mel.T,axis = 0)

    #8.Combining all the features
    combined = np.hstack([mfccs_mean,chroma_mean,contrast_mean,zcr_mean,rms_mean,mel_mean])

    return combined

def main():

    input_folder = r"F:\SEMESTER 5\PROJECTS\MACHINE LEARNING PROJECT\AUDIO DATASET 24"
    output_folder = r"F:\SEMESTER 5\PROJECTS\MACHINE LEARNING PROJECT\PREPROCESSED AUDIO DATASET 24"
    for root,subdirs,files in os.walk(input_folder):
        for file in files:
            if file.endswith(".wav"):
                in_path = os.path.join(root,file)
                rel_path = os.path.relpath(in_path,input_folder)
                out_path = os.path.join(output_folder,rel_path)
                preprocessing(in_path,out_path)

    
    input_folder = r"F:\SEMESTER 5\PROJECTS\MACHINE LEARNING PROJECT\PREPROCESSED AUDIO DATASET 24"
    output_file = r"F:\SEMESTER 5\PROJECTS\MACHINE LEARNING PROJECT\ravdess_dataset.csv"
    all_features = []
    all_emotions = []
    all_files = []

    for root,_,files in os.walk(input_folder):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root,file)
                features = extract_features(file_path)
                emotion = int(file.split('-')[2])

                all_features.append(features)
                all_emotions.append(emotion)
                all_files.append(file)


    #Converting the features,emotions files into a dataframe
    df = pd.DataFrame(all_features)
    df['file_name'] = all_files
    df['emotion_labels'] = all_emotions


    #Save as dataframe as CSV file
    df.to_csv(output_file,index = False)

main()