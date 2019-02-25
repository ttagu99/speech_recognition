#-*- coding: utf-8 -*-
import os
import re
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

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

# balancing imbalance data 
train_df      = train_df[train_df.label != 'silence']
train_unknown = train_df[train_df.label == 'unknown']
val_unknown = valid_df[valid_df.label == 'unknown']
train_df = train_df[train_df.label != 'unknown']
valid_df = valid_df[valid_df.label != 'unknown']

train_list = []
train_list.append(train_unknown)
for idx in range(18):
    train_list.append(train_df)

for idx in range(6000):
    train_list.append(silence_files)
train_df = pd.concat(train_list, ignore_index=True)

val_list = []
val_list.append(val_unknown)
for idx in range(16):
    val_list.append(valid_df)
for idx in range(700):
    val_list.append(silence_files)
valid_df = pd.concat(val_list, ignore_index=True)


train_pivot = train_df.pivot_table(index='label',aggfunc='count')
print('Train Data Check')
print(train_pivot)
valid_pivot = valid_df.pivot_table(index='label',aggfunc='count')
print('valid Data Check')
print(valid_pivot)

from scipy.io import wavfile

def read_wav_file(fname):
    _, wav = wavfile.read(fname)
    # normalize

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
    

    ret_wav = np.reshape(wav,(CL,1))

    specgram = stft(wav, CL, nperseg = 400, noverlap = 240, nfft = 512, padded = False, boundary = None)
    phase = np.angle(specgram[2]) / np.pi
    amp = np.log1p(np.abs(specgram[2]))
    
    return np.stack([phase, amp], axis = 2), ret_wav


import random
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, Activation, BatchNormalization, GlobalAveragePooling2D, GlobalMaxPool2D, concatenate, Dense, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPool1D
from tensorflow.python.keras.layers import SeparableConv2D
from tensorflow.python.keras.optimizers import RMSprop, SGD
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras import layers


def train_generator(train_batch_size):
    while True:
        this_train = train_df.groupby('label_id').apply(lambda x: x.sample(n = 2000))
        shuffled_ids = random.sample(range(this_train.shape[0]), this_train.shape[0])
        for start in range(0, len(shuffled_ids), train_batch_size):
            x_batch = []
            x_batch_1d = []
            y_batch = []
            end = min(start + train_batch_size, len(shuffled_ids))
            i_train_batch = shuffled_ids[start:end]
            for i in i_train_batch:
                x_2d, x_1d = process_wav_file(this_train.wav_file.values[i],phase='TRAIN')
                x_batch.append(x_2d)
                x_batch_1d.append(x_1d)
                y_batch.append(this_train.label_id.values[i])
            x_batch = np.array(x_batch)
            x_batch_1d = np.array(x_batch_1d)
            y_batch = to_categorical(y_batch, num_classes = len(POSSIBLE_LABELS))
            yield [x_batch, x_batch_1d] , y_batch


def valid_generator(val_batch_size):
    while True:
        ids = list(range(valid_df.shape[0]))
        for start in range(0, len(ids), val_batch_size):
            x_batch = []
            x_batch_1d = []
            y_batch = []
            end = min(start + val_batch_size, len(ids))
            i_val_batch = ids[start:end]
            for i in i_val_batch:
                x_2d, x_1d = process_wav_file(valid_df.wav_file.values[i],phase='TRAIN')
                x_batch.append(x_2d)
                x_batch_1d.append(x_1d)
                y_batch.append(valid_df.label_id.values[i])
            x_batch = np.array(x_batch)
            x_batch_1d = np.array(x_batch_1d)
            y_batch = to_categorical(y_batch, num_classes = len(POSSIBLE_LABELS))
            yield [x_batch, x_batch_1d], y_batch

from tensorflow.python.keras._impl.keras import backend as K

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    Arguments:
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    Returns:
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(
        filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2,
                                                                            2)):
    """conv_block is the block that has a conv layer at shortcut.

    Arguments:
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Tuple of integers.

    Returns:
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with
    strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(
        filters1, (1, 1), strides=strides,
        name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(
        filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(
        filters3, (1, 1), strides=strides,
        name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def identity_block_1d(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res_1d' + str(stage) + block + '_branch'
    bn_name_base = 'bn_1d' + str(stage) + block + '_branch'

    x = Conv1D(filters1, (1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization( name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv1D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv1D(filters3, (1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block_1d(input_tensor, kernel_size, filters, stage, block, strides=(2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res_1d' + str(stage) + block + '_branch'
    bn_name_base = 'bn_1d' + str(stage) + block + '_branch'

    x = Conv1D(filters1, (1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv1D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv1D(filters3, (1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = Conv1D(filters3, (1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


x_in_1d = Input(shape = (16000,1))
x_1d = BatchNormalization()(x_in_1d)

filter_num_1d =4
for i in range(9):
    x_1d = conv_block_1d(x_1d, 3, [filter_num_1d*(2 ** i), filter_num_1d*(2 ** i), (filter_num_1d*4)*(2 ** i)], stage=(i+1), block='a')
    x_1d = identity_block_1d(x_1d, 3, [filter_num_1d*(2 ** i), filter_num_1d*(2 ** i), (filter_num_1d*4)*(2 ** i)], stage=(i+1), block='b')
x_1d = Conv1D(32, (1))(x_1d)#128
x_branch_1_1d = GlobalAveragePooling1D()(x_1d)
x_branch_2_1d = GlobalMaxPool1D()(x_1d)
x_1d = concatenate([x_branch_1_1d, x_branch_2_1d])
x_1d = Dense(1024, activation = 'relu')(x_1d)
#x_1d = Dropout(0.2)(x_1d)

x_in = Input(shape = (257,98,2))
x = BatchNormalization()(x_in)
if K.image_data_format() == 'channels_last':
    bn_axis = 3
else:
    bn_axis = 1

x = Conv2D(16, (3, 3), strides=(1, 1), padding='same', name='conv1')(x)
x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

filter_num=16
x = conv_block(x, 3, [filter_num, filter_num, filter_num*4], stage=2, block='a', strides=(1, 1))
x = identity_block(x, 3, [filter_num, filter_num, filter_num*4], stage=2, block='b')
x = identity_block(x, 3, [filter_num, filter_num, filter_num*4], stage=2, block='c')

filter_num=filter_num*2
x = conv_block(x, 3, [filter_num, filter_num, filter_num*4], stage=3, block='a')
x = identity_block(x, 3, [filter_num, filter_num, filter_num*4], stage=3, block='b')
x = identity_block(x, 3, [filter_num, filter_num, filter_num*4], stage=3, block='c')
x = identity_block(x, 3, [filter_num, filter_num, filter_num*4], stage=3, block='d')

filter_num=filter_num*2
x = conv_block(x, 3, [filter_num, filter_num, filter_num*4], stage=4, block='a')
x = identity_block(x, 3, [filter_num, filter_num, filter_num*4], stage=4, block='b')
x = identity_block(x, 3, [filter_num, filter_num, filter_num*4], stage=4, block='c')
x = identity_block(x, 3, [filter_num, filter_num, filter_num*4], stage=4, block='d')
x = identity_block(x, 3, [filter_num, filter_num, filter_num*4], stage=4, block='e')
x = identity_block(x, 3, [filter_num, filter_num, filter_num*4], stage=4, block='f')

filter_num=filter_num*2
x = conv_block(x, 3, [filter_num, filter_num, filter_num*4], stage=5, block='a')
x = identity_block(x, 3, [filter_num, filter_num, filter_num*4], stage=5, block='b')
x = identity_block(x, 3, [filter_num, filter_num, filter_num*4], stage=5, block='c')
x = Conv2D(filter_num*4, (1,1))(x)
x_branch_1 = GlobalAveragePooling2D()(x)
x_branch_2 = GlobalMaxPool2D()(x)
x = concatenate([x_branch_1, x_branch_2])
x = Dense(1024, activation = 'relu')(x)
#x = Dropout(0.2)(x)
x_merge = concatenate([x, x_1d])
x_merge = Dropout(0.2)(x_merge)

x_merge = Dense(len(POSSIBLE_LABELS), activation = 'softmax')(x_merge)

fine_tune_weight = 'combi_reslast512_dropadd_2_sgd_simple1dres_silenceadd_balance.hdf5'
weight_name = 'combi_reslast512_lr01_drop1_sgd_simple1dres_silenceadd_balance.hdf5'
#weight_name = 'combi_reslast512_dropadd_sgd_simple1dres_silenceadd.hdf5'

model = Model(inputs = [x_in, x_in_1d], outputs = x_merge)

# FINE TUNE
model.load_weights(root_dir + 'weights/'+ fine_tune_weight)

opt = SGD(lr=0.001, momentum=0.9, decay=0.000001, nesterov=False)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

from keras_tqdm import TQDMCallback
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


batch_size = 64

callbacks = [EarlyStopping(monitor='val_loss',
                           patience=14,
                           verbose=1,
                           min_delta=0.00001,
                           mode='min'),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=7,
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
                              steps_per_epoch=int((train_df.shape[0]/batch_size)/18),#344,
                              epochs=100,
                              verbose=2,
                              callbacks=callbacks,
                              validation_data=valid_generator(batch_size),
                              validation_steps=int(np.ceil(valid_df.shape[0]/batch_size)))


model.load_weights(root_dir + 'weights/'+ weight_name)

test_paths = glob(os.path.join(root_dir , 'test/audio/*wav'))


def test_generator(test_batch_size):
    while True:
        for start in range(0, len(test_paths), test_batch_size):
            x_batch = []
            x_batch_1d = []
            end = min(start + test_batch_size, len(test_paths))
            this_paths = test_paths[start:end]
            for x in this_paths:
                x_2d, x_1d = process_wav_file(x,phase='TEST')
                x_batch.append(x_2d)
                x_batch_1d.append(x_1d)
            x_batch = np.array(x_batch)
            x_batch_1d = np.array(x_batch_1d)
            yield [x_batch, x_batch_1d]

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