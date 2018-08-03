import os
import errno

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


DATA_PATH = 'raw_data'

SAMPLE_RATE = 16000
DURATION = 2.5
OFFSET = 0.5
HOP_LENGTH = 512
# MFCC -> (n_mfcc, t)
# t = sample_rate * time / hop_length
MAX_LENGTH = int((SAMPLE_RATE * DURATION // HOP_LENGTH) + 1)

def preprocess_data():
    dir_lists = os.listdir(DATA_PATH)
    mfcc_vectors = []
    labels = []

    for dir_list in dir_lists:
        if dir_list == '.DS_Store':
            continue
        file_path = os.path.join(DATA_PATH, dir_list)
        files = os.listdir(file_path)
        print("==================== {} ====================".format(dir_list))
        for file in files:
            if file == '.DS_Store':
                continue
            label = get_label(file.strip('.wav'))
            mfcc = wav2mfcc(os.path.join(file_path, file), duration=DURATION, offset=OFFSET)
            print(file, mfcc.shape, label.shape)
            mfcc_vectors.append(mfcc)
            labels.append(label)
    
    mfcc_vectors = np.array(mfcc_vectors)
    labels = np.array(labels)    
    np.savez('train_data.npz', x_train=mfcc_vectors, y_train=labels)
    print(mfcc_vectors.shape, labels.shape)

def get_label(file_name):
    ''' Filename identifiers  

    Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
    Vocal channel (01 = speech, 02 = song).
    Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
    Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the ‘neutral’ emotion.
    Statement (01 = “Kids are talking by the door”, 02 = “Dogs are sitting by the door”).
    Repetition (01 = 1st repetition, 02 = 2nd repetition).
    Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
            
    '''
    file_name = file_name.split('-')
    label = []

    if int(file_name[6])%2 != 0: # male
        if file_name[2] == '01':
            label.append(0)
        elif file_name[2] == '02':
            label.append(1)
        elif file_name[2] == '03':
            label.append(2) 
        elif file_name[2] == '04':
            label.append(3) 
        elif file_name[2] == '05':
            label.append(4)
        elif file_name[2] == '06':
            label.append(5)
        elif file_name[2] == '07':
            label.append(6)
        elif file_name[2] == '08':
            label.append(7)
    else: # female
        if file_name[2] == '01':
            label.append(8)
        elif file_name[2] == '02':
            label.append(9)
        elif file_name[2] == '03':
            label.append(10) 
        elif file_name[2] == '04':
            label.append(11) 
        elif file_name[2] == '05':
            label.append(12)
        elif file_name[2] == '06':
            label.append(13)
        elif file_name[2] == '07':
            label.append(14)
        elif file_name[2] == '08':
            label.append(15)
    
    label = np.array(label)

    return label

def wav2mfcc(file_path, sr=None, offset=0.0, duration=None, n_mfcc=13, max_length=MAX_LENGTH):
    data, sr = librosa.load(file_path, mono=True, sr=sr, offset=offset, duration=duration)
    data = data[::3]
    mfcc = librosa.feature.mfcc(data, sr=16000, n_mfcc=n_mfcc)

    if (max_length > mfcc.shape[1]):
        #print(max_length, mfcc.shape[1])
        pad_width = max_length - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_length]
    
    '''
    # plot
    plt.figure()
    plt.subplot(2,1,1)
    librosa.display.waveplot(data, sr=sr)
    plt.subplot(2,1,2)
    librosa.display.specshow(mfcc, x_axis='time')
    #plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()
    '''

    return mfcc

def load_dataset(split_ratio=0.8, random_state=42):
    data = np.load('train_data.npz')
    x_train, y_train = data['x_train'], data['y_train']
    data.close()

    #y_train = np_utils.to_categorical(y_train, 16)
    return train_test_split(x_train, y_train, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)

def save_model(model, model_name):
    file_path = 'model/{}.h5'.format(model_name)
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    model.save(file_path)

def plot_loss(history, file_name):
    file_path = 'images/{}.png'.format(file_name)
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(file_path)
    plt.show()

def plot_accuracy(history, file_name):
    file_path = 'images/{}.png'.format(file_name)
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model train vs validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(file_path)
    plt.show()

if __name__ == "__main__":
    preprocess_data()

    #file_path = 'raw_data/Actor_08/03-01-08-01-02-01-08.wav'
    #file_name = '03-01-08-01-02-01-08'
    #mfcc = wav2mfcc(file_path, sr=None, offset=0.5, duration=2.5, n_mfcc=13)
    