#-*- coding: utf-8 -*-
import os
import re
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

POSSIBLE_LABELS = 'yes no up down left right on off stop go silence unknown'.split()
id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}
name2id = {name: i for i, name in id2name.items()}
root_dir = 'I:/imgfolder/voice/'

def load_data(data_dir):
    """ Return 2 lists of tuples:
    [(class_id, user_id, path), ...] for train
    [(class_id, user_id, path), ...] for validation
    """
    # Just a simple regexp for paths with three groups:
    # prefix, label, user_id
    pattern = re.compile("(.+\/)?(\w+)\/([^_]+)_.+wav")
    all_files = glob(os.path.join(data_dir, 'train/audio/*/*wav'))

    for idx, file in enumerate(all_files):
        all_files[idx] = file.replace('\\','/')

    with open(os.path.join(data_dir, 'train/validation_list.txt'), 'r') as fin:
        validation_files = fin.readlines()
    valset = set()
    for entry in validation_files:
        r = re.match(pattern, entry)
        if r:
            valset.add(r.group(3))

    possible = set(POSSIBLE_LABELS)
    train, val = [], []
    for entry in all_files:
        r = re.match(pattern, entry)
        if r:
            label, uid = r.group(2), r.group(3)
            if label == '_background_noise_':
                label = 'silence'
            if label not in possible:
                label = 'unknown'

            label_id = name2id[label]

            sample = (label, label_id, uid, entry)
            if uid in valset:
                val.append(sample)
            else:
                train.append(sample)

    print('There are {} train and {} val samples'.format(len(train), len(val)))
    
    columns_list = ['label', 'label_id', 'user_id', 'wav_file']
    
    train_df = pd.DataFrame(train, columns = columns_list)
    valid_df = pd.DataFrame(val, columns = columns_list)
    
    return train_df, valid_df


train_df, valid_df = load_data(root_dir)

silence_files = train_df[train_df.label == 'silence']
train_df      = train_df[train_df.label != 'silence']



from scipy.io import wavfile

def read_wav_file(fname):
    _, wav = wavfile.read(fname)
    wav = wav.astype(np.float32) / np.iinfo(np.int16).max
    return wav

silence_data = np.concatenate([read_wav_file(x) for x in silence_files.wav_file.values])


from scipy.signal import stft

def process_wav_file(fname, phase):
    wav = read_wav_file(fname)

        # time streching
    if phase == 'TRAIN':
        time_strech_flag = np.random.randint(2)
        if time_strech_flag == 1:
            ts_ratio = np.random.uniform(0.8,1.2)     
            wav = np.interp(np.arange(0, len(wav), ts_ratio), np.arange(0, len(wav)), wav)

    L = 19200  # 1 sec
    CL = 16000 # crop 
    if phase == 'TRAIN' :
        if len(wav) > L:
            i = np.random.randint(0, len(wav) - L)
            wav = wav[i:(i+L)]
        elif len(wav) < L:
            rem_len = L - len(wav)
            i = np.random.randint(0, len(silence_data) - rem_len)
            silence_part = silence_data[i:(i+L)]
            j = np.random.randint(0, rem_len)
            silence_part_left  = silence_part[0:j]
            silence_part_right = silence_part[j:rem_len]
            wav = np.concatenate([silence_part_left, wav, silence_part_right])
    else:
        if len(wav) > L: #center crop
            i =  int((len(wav) - L)/2)
            wav = wav[i:(i+L)]
        elif len(wav) < L: #silence add side
            rem_len = L - len(wav)
            i = int((len(silence_data) - rem_len)/2)
            silence_part = silence_data[i:(i+L)]
            j = int(rem_len/2)
            silence_part_left  = silence_part[0:j]
            silence_part_right = silence_part[j:rem_len]
            wav = np.concatenate([silence_part_left, wav, silence_part_right])

    # crop
    if phase == 'TRAIN':
        i = np.random.randint(0,L-CL)
        wav = wav[i:(i+CL)]
    else:
        i = int((L-CL)/2)
        wav = wav[i:(i+CL)]

    # nosise add
    if phase == 'TRAIN':
        noise_add_flag = np.random.randint(2)
        if noise_add_flag == 1:
            noise_ratio = np.random.uniform(0.0,0.5)
            i = np.random.randint(0, len(silence_data) - CL)
            silence_part = silence_data[i:(i+CL)]
            org_max = max(wav)
            silence_max = max(silence_part)
            silence_part = silence_part * (org_max/silence_max)
            wav = wav*(1.0-noise_ratio) + silence_part * noise_ratio
    

    #ret_wav = np.reshape(wav,(CL,1))

    specgram = stft(wav, CL, nperseg = 400, noverlap = 240, nfft = 512, padded = False, boundary = None)
    phase = np.angle(specgram[2]) / np.pi
    amp = np.log1p(np.abs(specgram[2]))
    
    return np.stack([phase, amp,  phase], axis = 2)


import random
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, Activation, BatchNormalization, GlobalAveragePooling2D, GlobalMaxPool2D, concatenate, Dense, Dropout
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.utils import to_categorical


def train_generator(train_batch_size):
    while True:
        this_train = train_df.groupby('label_id').apply(lambda x: x.sample(n = 2000))
        shuffled_ids = random.sample(range(this_train.shape[0]), this_train.shape[0])
        for start in range(0, len(shuffled_ids), train_batch_size):
            x_batch = []
            y_batch = []
            end = min(start + train_batch_size, len(shuffled_ids))
            i_train_batch = shuffled_ids[start:end]
            for i in i_train_batch:
                x_batch.append(process_wav_file(this_train.wav_file.values[i],phase='TRAIN'))
                y_batch.append(this_train.label_id.values[i])
            x_batch = np.array(x_batch)
            y_batch = to_categorical(y_batch, num_classes = len(POSSIBLE_LABELS))
            yield x_batch, y_batch


def valid_generator(val_batch_size):
    while True:
        ids = list(range(valid_df.shape[0]))
        for start in range(0, len(ids), val_batch_size):
            x_batch = []
            y_batch = []
            end = min(start + val_batch_size, len(ids))
            i_val_batch = ids[start:end]
            for i in i_val_batch:
                x_batch.append(process_wav_file(valid_df.wav_file.values[i],phase='TRAIN'))
                y_batch.append(valid_df.label_id.values[i])
            x_batch = np.array(x_batch)
            y_batch = to_categorical(y_batch, num_classes = len(POSSIBLE_LABELS))
            yield x_batch, y_batch

from keras.applications.xception import Xception

model = Xception(include_top = True, weights = None,input_shape=(257,98,3),classes = len(POSSIBLE_LABELS))

from keras.optimizers import SGD
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


from keras_tqdm import TQDMCallback
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

weight_name = 'xcetion_aug_fine.hdf5'
batch_size = 1

callbacks = [EarlyStopping(monitor='val_loss',
                           patience=6,
                           verbose=1,
                           min_delta=0.00001,
                           mode='min'),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=5,
                               verbose=1,
                               epsilon=0.0001,
                               mode='min'),
             ModelCheckpoint(monitor='val_loss',
                             filepath=root_dir + 'weights/' + weight_name,
                             save_best_only=True,
                             save_weights_only=True,
                             mode='min') ,
             TQDMCallback()]

history = model.fit_generator(generator=train_generator(batch_size),
                              steps_per_epoch=1,#int(np.ceil(train_df.shape[0]/batch_size)/300),#344,
                              epochs=60,
                              verbose=2,
                              callbacks=callbacks,
                              validation_data=valid_generator(batch_size),
                              validation_steps=1)#int(np.ceil(valid_df.shape[0]/batch_size))*20)


model.load_weights(root_dir + 'weights/'+ weight_name)

test_paths = glob(os.path.join(root_dir , 'test/audio/*wav'))


def test_generator(test_batch_size):
    while True:
        for start in range(0, len(test_paths), test_batch_size):
            x_batch = []
            end = min(start + test_batch_size, len(test_paths))
            this_paths = test_paths[start:end]
            for x in this_paths:
                x_batch.append(process_wav_file(x,phase='TEST'))
            x_batch = np.array(x_batch)
            yield x_batch

predictions = model.predict_generator(test_generator(batch_size), int(np.ceil(len(test_paths)/batch_size)))
classes = np.argmax(predictions, axis=1)

# last batch will contain padding, so remove duplicates
submission = dict()
for i in range(len(test_paths)):
    fname, label = os.path.basename(test_paths[i]), id2name[classes[i]]
    submission[fname] = label


with open(root_dir + weight_name + '.csv', 'w') as fout:
    fout.write('fname,label\n')
    for fname, label in submission.items():
        fout.write('{},{}\n'.format(fname, label))