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

import random
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv1D, AvgPool1D, MaxPooling1D, Activation, BatchNormalization, GlobalAveragePooling1D, GlobalMaxPool1D, concatenate, Dense, Dropout
from tensorflow.python.keras.optimizers import RMSprop, SGD
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras import layers
from tensorflow.python.keras._impl.keras import backend as K
from keras.utils.training_utils import multi_gpu_model
from keras.utils.generic_utils import get_custom_objects
from keras.models import Sequential

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
used_train_df = train_df[train_df.label != 'unknown']
used_valid_df = valid_df[valid_df.label != 'unknown']

train_list = []
train_list.append(train_unknown)
for idx in range(18):
    train_list.append(used_train_df)

for idx in range(6000):
    train_list.append(silence_files)
train_df = pd.concat(train_list, ignore_index=True)

val_list = []
val_list.append(val_unknown)
for idx in range(16):
    val_list.append(used_valid_df)
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

def normalize_wav(wav):
    wav_mean = np.mean(wav)
    wav = wav - wav_mean
    wav_max = max(abs(wav))
    if wav_max == 0 : # zero divide error
        wav_max = 0.01
    wav = wav.astype(np.float32)/wav_max
    return wav

def read_wav_file(fname):
    _, wav = wavfile.read(fname)
    wav = normalize_wav(wav)
    #wav = wav.astype(np.float32) / np.iinfo(np.int16).max
    return wav

def pre_emphasis(wav):
    pre_emphasis = np.random.uniform(0.95,0.97)
    ret_wav = np.append(wav[0], wav[1:] - pre_emphasis * wav[:-1])
    wav_max = max(abs(ret_wav))
    ret_wav = ret_wav/wav_max
    return ret_wav

def swish(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'swish': Activation(swish)})

silence_data = np.concatenate([read_wav_file(x) for x in silence_files.wav_file.values])


from scipy.signal import stft

def center_align_resize(wav, length):
    if len(wav) > length: #center crop
        i =  int((len(wav) - length)/2)
        wav = wav[i:(i+length)]
    elif len(wav) < length: #silence add side
        rem_len = length - len(wav)
        i = int((len(silence_data) - rem_len)/2)
        silence_part = silence_data[i:(i+length)]
        j = int(rem_len/2)
        silence_part_left  = silence_part[0:j]
        silence_part_right = silence_part[j:rem_len]
        wav = np.concatenate([silence_part_left, wav, silence_part_right])
    return wav

def process_wav_file(fname, phase, dim='1D', optlabel = None):
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
        center_align_resize(wav,L)

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

    if phase == 'TRAIN':
        white_noise_add_flag = np.random.randint(2)
        if white_noise_add_flag == 1:
            wn_ratio = np.random.uniform(0.0,0.1)
            wn = np.random.randn(len(wav))
            wav = wav + wn_ratio*wn

    #if phase == 'TRAIN':
    #    pre_emphasis_flag = np.random.randint(2)
    #    if pre_emphasis_flag == 1:
    #        wav = pre_emphasis(wav)

    # crop
    if phase == 'TRAIN' and optlabel != None and optlabel != 'silence':
        add_audio_flag = np.random.randint(2)
        if add_audio_flag == 1:
            type = np.random.randint(3)
            if type == 0 : # equal label 
                if optlabel == 'unknown': # unkown + unknown
                    add_wav_file = train_unknown.sample().wav_file.values[0]
                    add_wav = read_wav_file(add_wav_file)
                    add_wav = center_align_resize(add_wav,CL)
                    wav = wav + add_wav
                else:
                    add_wav_file = used_train_df[used_train_df.label == optlabel].sample().wav_file.values[0] # used + equal used
                    add_wav = read_wav_file(add_wav_file)
                    add_wav = center_align_resize(add_wav,CL)
                    wav = wav + add_wav
            if type ==1 : # used but un equal label -> unkown or ratio depend
                add_wav_file = used_train_df.sample().wav_file.values[0]
                add_wav = read_wav_file(add_wav_file)
                add_wav = center_align_resize(add_wav,CL)
                mix_ratio = np.random.uniform(0.0,0.1)
                wav = wav *(1-mix_ratio) + add_wav * mix_ratio
            if type ==2: # add unkown label -> equal label  used + unkown ratio
                add_wav_file = train_unknown.sample().wav_file.values[0]
                add_wav = read_wav_file(add_wav_file)
                add_wav = center_align_resize(add_wav,CL)
                mix_ratio = np.random.uniform(0.1,0.4)
                wav = wav *(1-mix_ratio) + add_wav * mix_ratio

    wav = normalize_wav(wav)
    #return np.stack([phase, amp], axis = 2)
    if dim=='1D':
        ret_wav = np.reshape(wav,(CL,1))
        return ret_wav
    elif  dim == '2D':
        specgram = stft(wav, CL, nperseg = 400, noverlap = 240, nfft = 512, padded = False, boundary = None)
        phase = np.angle(specgram[2]) / np.pi
        amp = np.log1p(np.abs(specgram[2]))
        return np.stack([phase, amp], axis = 2)
    else : # combi
        ret_wav = np.reshape(wav,(CL,1))
        specgram = stft(wav, CL, nperseg = 400, noverlap = 240, nfft = 512, padded = False, boundary = None)
        phase = np.angle(specgram[2]) / np.pi
        amp = np.log1p(np.abs(specgram[2]))
        return np.stack([phase, amp], axis = 2), ret_wav



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
                x_batch.append(process_wav_file(this_train.wav_file.values[i],phase='TRAIN', optlabel =this_train.label.values[i]))
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

if K.image_data_format() == 'channels_last':
    bn_axis = 3
else:
    bn_axis = 1

init_filter_num = 8
x_in_1d = Input(shape = (16000,1))
x_1d = BatchNormalization(name = 'batchnormal_1d_in')(x_in_1d)
for i in range(9):
    name = 'step'+str(i)
    x_1d = Conv1D(init_filter_num*(2 ** i), (3),padding = 'same', name = 'conv'+name+'_1')(x_1d)
    x_1d = BatchNormalization(name = 'batch'+name+'_1')(x_1d)
    if i !=0:
        x_1d = layers.add([x_1d_concate, x_1d])
    x_1d = Activation('relu')(x_1d)
    for j in range(2,11):
        short_cut = x_1d
        x_1d = Conv1D(init_filter_num*(2 ** i), (3),padding = 'same', name = 'conv'+name+'_'+str(j))(x_1d)
        x_1d = BatchNormalization(name = 'batch'+name+'_'+str(j))(x_1d)
        x_1d = layers.add([short_cut,x_1d])
        x_1d = Activation('relu')(x_1d)
    x_1d_max = MaxPooling1D((2), padding='same')(x_1d)
    x_1d_avg = AvgPool1D((2), padding='same')(x_1d)
    x_1d = layers.add([x_1d_max, x_1d_avg])
    x_1d = BatchNormalization(name = 'batch'+name+'_avgmax_add')(x_1d)
    x_1d = Activation('relu')(x_1d)
    if i != 8:
        x_1d_concate = layers.concatenate([x_1d_max, x_1d_avg])
        x_1d_concate = BatchNormalization(name = 'batch'+name+'_avgmax_concate')(x_1d_concate)
x_1d = Conv1D(2048, (1),name='last2048')(x_1d)#128
x_1d_branch_1 = GlobalAveragePooling1D()(x_1d)
x_1d_branch_2 = GlobalMaxPool1D()(x_1d)
x_1d = concatenate([x_1d_branch_1, x_1d_branch_2])
x_1d = Dense(1024, activation = 'relu', name= 'dense2048_1024')(x_1d)
x_1d = Dropout(0.2)(x_1d)
x_1d = Dense(len(POSSIBLE_LABELS), activation = 'softmax',name='cls_1d')(x_1d)

fine_tune_weight = '1dcnn_last1024_noiseadd_ts_mixset_10res_allcon_balance_inputnormal_submean_abs_whitenadd_sgd.hdf5'
weight_name = '1dcnn_last2048_noiseadd_ts_mixset_10res_allcon_balance_inputnormal_submean_abs_whitenadd_sgd.hdf5'
#weight_name = 'combi_reslast512_dropadd_sgd_simple1dres_silenceadd.hdf5'


# the results from the gradient updates on the CPU
#with tf.device("/cpu:0"):
model = Model(inputs = x_in_1d, outputs = x_1d)
# FINE TUNE
model.load_weights(root_dir + 'weights/'+ fine_tune_weight, by_name=True)
#model = multi_gpu_model(model, gpus=2)
opt = SGD(lr = 0.01, momentum = 0.9, decay = 0.000001,nesterov=True)
#opt = RMSprop(lr = 0.00001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

from keras_tqdm import TQDMCallback
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard


batch_size = 32

callbacks = [EarlyStopping(monitor='val_loss',
                           patience=16,
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
             TQDMCallback(),
             TensorBoard(log_dir=root_dir+ weight_name.split('.')[0], histogram_freq=0, write_graph=True, write_images=True)             
             ]


history = model.fit_generator(generator=train_generator(batch_size),
                              steps_per_epoch=int((train_df.shape[0]/batch_size)/18),#344,
                              epochs=100,
                              verbose=2,
                              callbacks=callbacks,
                              validation_data=valid_generator(batch_size),
                              validation_steps=int(np.ceil(valid_df.shape[0]/batch_size)))


model.load_weights(root_dir + 'weights/'+ weight_name)

test_paths = glob(os.path.join(root_dir , 'test/audio/*wav'))



def get_test_set_1d(path, tta=1):
    if tta ==1:
        x_batch = []
        x_batch.append(process_wav_file(path,phase='TEST',dim='1D'))
        x_batch = np.array(x_batch)
        return x_batch

def get_test_set_2d(path, tta=1):
    if tta ==1:
        x_batch = []
        x_batch.append(process_wav_file(path,phase='TEST',dim='2D'))
        x_batch = np.array(x_batch)
        return x_batch

def get_test_set_combi(path, tta=1):
    if tta ==1:
        x_batch = []
        x_batch_1d = []
        x2d, x1d = process_wav_file(path,phase='TEST',dim='combi')
        x_batch.append(x2d)
        x_batch_1d.append(x1d)
        x_batch = np.array(x_batch)
        x_batch_1d = np.array(x_batch_1d)
        return [x_batch, x_batch_1d]

subfile = open(root_dir + weight_name +'_sub'+ '.csv', 'w')
probfile = open(root_dir + weight_name +'_prob'+ '.csv', 'w')
subfile.write('fname,label\n')
probfile.write('fname,yes,no,up,down,left,right,on,off,stop,go,silence,unknown\n')

for idx, path in enumerate(test_paths):
    fname = path.split('\\')[-1]
    probs = model.predict(get_test_set_1d(path),batch_size=1)
    label = id2name[np.argmax(probs)]
    subfile.write('{},{}\n'.format(fname,label))
    probfile.write(fname+',')
    print (str(idx) +'/' + str(len(test_paths)))
    for p, prob in enumerate(probs[0]):
        probfile.write(str(prob))
        if p == 11:
            probfile.write('\n')
        else:
            probfile.write(',')
