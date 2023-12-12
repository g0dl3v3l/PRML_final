from __future__ import print_function

import os,sys
import numpy as np
import scipy.io as scio
import tensorflow as tf
import keras
from keras.layers import Input, GRU, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, TimeDistributed, LSTM
from keras.models import Model, load_model
import keras.backend as K
from sklearn.metrics import confusion_matrix
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from innvestigate import create_analyzer
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
# Parameters
use_existing_model = False
fraction_for_test = 0.1
data_dir = 'BVP/'
ALL_MOTION = [1, 2, 3, 4, 5, 6]
N_MOTION = len(ALL_MOTION)
T_MAX = 38
n_epochs = 25
f_dropout_ratio = 0.5
n_gru_hidden_units = 128
n_lstm_hidden_units = 128
n_batch_size = 64
f_learning_rate = 0.001
lrp_visualization_file_template = 'lrp_visualization_orientation_{}.png'



def normalize_data(data_1):
    # data(ndarray)=>data_norm(ndarray): [20,20,T]=>[20,20,T]
    data_1_max = np.concatenate((data_1.max(axis=0), data_1.max(axis=1)), axis=0).max(axis=0)
    data_1_min = np.concatenate((data_1.min(axis=0), data_1.min(axis=1)), axis=0).min(axis=0)
    if len(np.where((data_1_max - data_1_min) == 0)[0]) > 0:
        return data_1
    data_1_max_rep = np.tile(data_1_max, (data_1.shape[0], data_1.shape[1], 1))
    data_1_min_rep = np.tile(data_1_min, (data_1.shape[0], data_1.shape[1], 1))
    data_1_norm = (data_1 - data_1_min_rep) / (data_1_max_rep - data_1_min_rep)
    return data_1_norm

def zero_padding(data, T_MAX):
    # data(list)=>data_pad(ndarray): [20,20,T1/T2/...]=>[20,20,T_MAX]
    data_pad = []
    for i in range(len(data)):
        t = np.array(data[i]).shape[2]
        data_pad.append(np.pad(data[i], ((0, 0), (0, 0), (0, T_MAX - t)), 'constant', constant_values=0).tolist())
        print('Zero-padding: ' + str(i) + '/' + str(len(data)) + '\n')
    return np.array(data_pad) 
def onehot_encoding(label, num_class):
    # label(list)=>_label(ndarray): [N,]=>[N,num_class]
    label = np.array(label).astype('int32')
    # assert (np.arange(0,np.unique(label).size)==np.unique(label)).prod()    # Check label from 0 to N
    label = np.squeeze(label)
    _label = np.eye(num_class)[label-1]     # from label to onehot
    return _label

def leave_outOne(path_to_data, motion_sel,T_MAX_ORIGINAL, target_orientation=None , ):
    global T_MAX
    data = []
    label = []
    i =0
    T_MAX_ORIGINAL = T_MAX
    for data_root, data_dirs, data_files in os.walk(path_to_data):
        for data_file_name in data_files:

            file_path = os.path.join(data_root, data_file_name)
            try:
                data_1 = scio.loadmat(file_path)['velocity_spectrum_ro']
                label_1 = int(data_file_name.split('-')[1])
                location = int(data_file_name.split('-')[2])
                orientation = int(data_file_name.split('-')[3])
                repetition = int(data_file_name.split('-')[4])

                # Select Motion
                if label_1 not in motion_sel:
                    continue

                # Select Location
                # if (location not in [1,2,3,5]):
                #     continue

                # Select Orientation
                # Keep the selected orientation as the target, skip others
                if target_orientation is not None and orientation != target_orientation:
                    continue

                print(file_path + '\n')
                data_normed_1 = normalize_data(data_1)
                if T_MAX < np.array(data_1).shape[2]:
                    break   
                    T_MAX = np.array(data_1).shape[2]
                                 
                

            except Exception:
                continue

            data.append(data_normed_1.tolist())
            label.append(label_1)
            # i = i + 1
            # if i >= 1000:
            #     break
            
           

    print('Zero-padding...'+'\n')
    data = zero_padding(data, T_MAX_ORIGINAL)
    print('Done! zero padding\n')

    print('Swapping axes...'+'\n')
    data = np.swapaxes(np.swapaxes(data, 1, 3), 2, 3)
    data = np.expand_dims(data, axis=-1)
    print('Done! Swapping axes\n')

    print('Converting label to ndarray...'+'\n')
    label = np.array(label)
    print('Done! Converting label to ndarray\n')

    return data, label


def load_data(path_to_data, motion_sel, target_orientation=None):
    global T_MAX
    data = []
    label = []
    i = 0
    T_MAX = 0
    for data_root, data_dirs, data_files in os.walk(path_to_data):
        for data_file_name in data_files:
           
            file_path = os.path.join(data_root,data_file_name)
            try:
                data_1 = scio.loadmat(file_path)['velocity_spectrum_ro']
                label_1 = int(data_file_name.split('-')[1])
                location = int(data_file_name.split('-')[2])
                orientation = int(data_file_name.split('-')[3])
                repetition = int(data_file_name.split('-')[4])

                # Select Motion
                if (label_1 not in motion_sel):
                    continue

                #Select Location
                # if (location not in [1,2,3,5]):
                #     continue

                # Select Orientation
                # if (orientation not in [1,2,4,5]):
                #     continue
                
                # Normalization
                print(file_path + '\n')
                data_normed_1 = normalize_data(data_1)
                
                # Update T_MAX
                if T_MAX < np.array(data_1).shape[2]:
                    T_MAX = np.array(data_1).shape[2]  
                
                              
            except Exception:
                continue

            # Save List
            data.append(data_normed_1.tolist())
            label.append(label_1)
            # i = i + 1
            # if i >= 1000:
            #     break
            
            
    # Zero-padding
    print('Zero-padding...'+'\n')
    data = zero_padding(data, T_MAX)
    print('Done! zero padding\n')

    # Swap axes
    print('Swapping axes...'+'\n')

    data = np.swapaxes(np.swapaxes(data, 1, 3), 2, 3)   # [N,20,20',T_MAX]=>[N,T_MAX,20,20']
    data = np.expand_dims(data, axis=-1)    # [N,T_MAX,20,20]=>[N,T_MAX,20,20,1]
    print('Done! Swapping axes\n')
    # Convert label to ndarray
    print('Converting label to ndarray...'+'\n')
    label = np.array(label)
    print('Done! Converting label to ndarray\n')
    # data(ndarray): [N,T_MAX,20,20,1], label(ndarray): [N,N_MOTION]
    print('Done!\n')
    return data, label


def assemble_model(input_shape, n_class):
    model_input = Input(shape=input_shape, dtype='float32', name='name_model_input')

    x = TimeDistributed(Conv2D(16, kernel_size=(5,5), activation='relu', data_format='channels_last', \
        input_shape=input_shape))(model_input)
    x = TimeDistributed(MaxPooling2D(pool_size=(2,2)))(x)
    x = TimeDistributed(Flatten())(x)
    x = TimeDistributed(Dense(64, activation='relu'))(x)
    x = TimeDistributed(Dropout(f_dropout_ratio))(x)
    x = TimeDistributed(Dense(64, activation='relu'))(x)
    x = GRU(n_gru_hidden_units, return_sequences=False)(x)
   # x = LSTM(n_lstm_hidden_units, return_sequences=False)(x)
    x = Dropout(f_dropout_ratio)(x)
    model_output = Dense(n_class, activation='softmax', name='name_model_output')(x)

    model = Model(inputs=model_input, outputs=model_output)
    model.compile(optimizer=keras.optimizers.legacy.RMSprop(learning_rate=f_learning_rate),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
    return model


# Let's BEGIN >>>>
if len(sys.argv) < 2:
    print('Please specify GPU ...')
    exit(0)
if sys.argv[1] == '1' or sys.argv[1] == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
else:
    print('Wrong GPU number, 0 or 1 supported!')
    exit(0)


# ... (previous code)

# Leave-One-Out Cross-Validation Loop


orientations_for_cv = [1, 2, 4, 5]
for target_orientation in orientations_for_cv:
    # Load data for the current orientation
    data, label = leave_outOne(data_dir, ALL_MOTION, target_orientation)

    # Perform train-test split
    [data_train, data_test, label_train, label_test] = train_test_split(data, label, test_size=fraction_for_test)
    print('\nTrain on ' + str(label_train.shape[0]) + ' samples\n' +\
        'Test on ' + str(label_test.shape[0]) + ' samples\n')

    # One-hot encoding for train data
    label_train = onehot_encoding(label_train, N_MOTION)

    # Load existing model
    model = load_model('model_widar3_trained.h5')
    modified_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    model.summary()

    # Testing...
    print('Testing...')
    label_test_pred = model.predict(data_test)
    label_test_pred = np.argmax(label_test_pred, axis=-1) + 1

    # Confusion Matrix
    cm = confusion_matrix(label_test, label_test_pred)
    print(cm)
    epsilon = 1e-15
    cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis ] + epsilon)
    cm = np.around(cm, decimals=2)
    print(cm)

    # Plotting Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=ALL_MOTION, yticklabels=ALL_MOTION)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f'confusion_matrix_orientation_{target_orientation}.png')  # Save the figure as a PNG file
    # save_lrp_visualization(modified_model, data_test, label_test, target_orientation)
