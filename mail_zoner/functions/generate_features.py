import gc

from transformers import TFXLMRobertaModel
import tensorflow as tf
import numpy as np
import tqdm

from ..util.util import onehot
from ..parameters import MODELNAME


def encoder_xlmroberta():
    xlm_roberta = TFXLMRobertaModel.from_pretrained(MODELNAME,output_hidden_states=True)

    # Freeze weights
    xlm_roberta.trainable=False

    # Inputs
    word_ids_inp = tf.keras.layers.Input(shape=(512,), dtype=tf.int32,name="input_ids")
    attention_mask_inp = tf.keras.layers.Input(shape=(512,), dtype=tf.int32,name="attention_mask")

    # Transformer
    hidden_states = xlm_roberta([word_ids_inp,attention_mask_inp])[2]

    # Last 4 hidden states
    state_1_rev = hidden_states[-1]
    state_2_rev = hidden_states[-2]
    state_3_rev = hidden_states[-3]
    state_4_rev = hidden_states[-4]
    
    # Concatenation
    hidden_states_conc = tf.keras.layers.Concatenate(axis=-1)([state_1_rev,state_2_rev,state_3_rev,state_4_rev])

    # Feature Vector by averaging
    features = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_last",keepdims=True)(hidden_states_conc)
    features_flat = tf.keras.layers.Flatten()(features)

    encoder = tf.keras.Model(inputs=[word_ids_inp,attention_mask_inp],outputs=features_flat)

    return encoder


def save_features(model,dataset_tokenized,savedir,subfolder='train'):
    print(10*"*")
    print(f"Saving {subfolder} features")
    print(10*"*")
    
    for key, val in tqdm.tqdm(dataset_tokenized.items()):
        
        savedir_subfolder =  savedir / subfolder

        input_ids = val['tokenized']['input_ids']
        attention_mask = val['tokenized']['attention_mask']

        feature = model.predict([input_ids,attention_mask])
        label_onehot = onehot(val['labels_encoded'])

        savepath_features = savedir_subfolder.joinpath(f"{key}.npy")
        savepath_labels = savedir_subfolder.joinpath(f"{key}_l.npy")

        
        with open(savepath_features, 'wb') as f:
            np.save(f, feature)

        with open(savepath_labels, 'wb') as g:
            np.save(g, label_onehot)

        gc.collect()