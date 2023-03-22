import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow_io as tfio

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def spect_augment_set(data_dir, set_name, param_masking=30):
    
    # Collect files to augment
    aug_dir = os.path.join(data_dir, set_name, 'augmented')
    files_regex = os.path.join(aug_dir, r'*.wav')
    files = glob.glob(files_regex)
    
    # Create directory for melspectrograms
    mels_path = os.path.join(data_dir, set_name, 'melspec')
    make_dir(mels_path)
    
    # Path to save labels
    labels_path = os.path.join(data_dir, set_name, f'{set_name}_labels.csv')
    
    y = []
    count = 0
    meanSignalLength = 156027
    for fn in tqdm(files):
        label = os.path.splitext(os.path.basename(fn))[0].split('_')[1]
        signal , sr = librosa.load(fn)
        s_len = len(signal)
        
        # Add zero padding to the signal if less than 156027 (~4.07 seconds)
        if s_len < meanSignalLength:
               pad_len = meanSignalLength - s_len
               pad_rem = pad_len % 2
               pad_len //= 2
               signal = np.pad(signal, (pad_len, pad_len + pad_rem), 'constant', constant_values=0)
        
        # Remove from begining and the end if signal length is greater than 156027 (~4.07 seconds)
        else:
               pad_len = s_len - meanSignalLength
               pad_len //= 2
               signal = signal[pad_len:pad_len + meanSignalLength]

        mel_spectrogram = librosa.feature.melspectrogram(y=signal,
                                                         sr=sr,
                                                         n_mels=128,
                                                         hop_length=512,
                                                         fmax=8000,
                                                         n_fft=512,
                                                         center=True)
        
        dbscale_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max, top_db=80)
        img = plt.imshow(dbscale_mel_spectrogram, interpolation='nearest',origin='lower')
        plt.axis('off')
        plt.savefig(os.path.join(mels_path, f'{count}.png'), bbox_inches='tight')
        plt.close('all')
        count+=1
        
        y.append(label)
        if label == '1': # if COVID-19
            freq_mask = tfio.audio.freq_mask(dbscale_mel_spectrogram, param=param_masking)
            time_mask = tfio.audio.time_mask(freq_mask, param=param_masking)
            img = plt.imshow(time_mask,origin='lower')
            plt.axis('off')
            plt.savefig(os.path.join(mels_path, f'{count}.png'), bbox_inches='tight')
            plt.close('all')
            count+=1
            y.append(label)
        
        freq_mask = tfio.audio.freq_mask(dbscale_mel_spectrogram, param=param_masking)
        time_mask = tfio.audio.time_mask(freq_mask, param=param_masking)
        img = plt.imshow(time_mask,origin='lower')
        plt.axis('off')
        plt.savefig(os.path.join(mels_path, f'{count}.png'), bbox_inches='tight')
        plt.close('all')
        count+=1
        y.append(label)
    
    # Save labels
    y = pd.DataFrame(data={'label': y})
    y.to_csv(labels_path, index=False)
    
if __name__ == '__main__':
    file_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(file_dir, 'coughvid_data')
    
    spect_augment_set(data_dir, 'train')
    spect_augment_set(data_dir, 'valid')
    spect_augment_set(data_dir, 'test')