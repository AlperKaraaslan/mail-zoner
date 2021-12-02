import random
import math

import tensorflow as tf
import numpy as np

from ..parameters import MAXTIMESTEP, MASKVALUE, BATCHSIZE


class DatasetGenerator(tf.keras.utils.Sequence):
    def __init__(self,data_path,batch_size=BATCHSIZE,shuffle=True):
        self.data_path = data_path
        self.data_path_glob = data_path.glob('**/*.npy') #complete path to all samples (including labels _l.npy)
        
        self.ids_uniques = self.__list_ids() #list of all sample ids

        # Variables
        self.batch_size = batch_size
        
        # Configurations
        self.shuffle = shuffle
        self.on_epoch_end()

        
    def __len__(self):
        # Number of batches per epoch
        return math.ceil(len(self.ids_uniques)/self.batch_size)


    def __list_ids(self):
        '''
        returns list of all unique ids in folder "data_path"
        '''
        data_path_glob = list(self.data_path_glob)

        elements = sorted([i.stem for i in data_path_glob])
        ids_list = list(set([i.split("_")[0] for i in elements]))

        return ids_list


    def __get_sample(self,identifier):
        data_path = self.data_path
        
        feature_path = data_path.joinpath(f"{identifier}.npy") #feature path to identifier
        label_path = data_path.joinpath(f"{identifier}_l.npy") #label path to identifier
        
        feature = np.load(feature_path) #load feature
        label_onehot = np.load(label_path) #load label

        feature = np.expand_dims(feature, axis=0) #expand dimension to concatenate later with other samples
        label_onehot = np.expand_dims(label_onehot, axis=0)

        feature_pad,label_onehot_pad = self.__pad(feature,label_onehot) #pad to 
        
        return feature_pad,label_onehot_pad


    def __getitem__(self, index):
        batch_size = self.batch_size
        
        identifier_list = self.ids_uniques[index*batch_size : (index + 1) *batch_size]
        
        features_list = [self.__get_sample(i)[0] for i in identifier_list]
        labels_list = [self.__get_sample(i)[1] for i in identifier_list]
        
        features_batch = np.concatenate(features_list,axis=0)        
        labels_batch = np.concatenate(labels_list,axis=0)  
        
        return features_batch,labels_batch
        
            
    def __pad(self,x,y): #batch size can be tricke as pad_seq is padding in axis=1!
        x_padded = tf.keras.preprocessing.sequence.pad_sequences(
            x, maxlen=MAXTIMESTEP, dtype="float32", padding="post", value=MASKVALUE
            )

        y_padded = tf.keras.preprocessing.sequence.pad_sequences(
            y, maxlen=MAXTIMESTEP, dtype="float32", padding="post", value=MASKVALUE
            )

        return x_padded,y_padded        


    def on_epoch_end(self):
        # After each epoch, rearranges indices
        if self.shuffle:
            random.shuffle(self.ids_uniques)