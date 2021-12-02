import pathlib
import functools

from transformers import AutoTokenizer
import numpy as np
import tensorflow as tf
import pandas as pd

from ..parameters import TOKENIZER,MAXLENMAIL,PAD,CHECKPOINTNAME,MAXTIMESTEP,MASKVALUE,LABELMAPPINGS_rev
from ..functions.model import segmenter
from ..functions.generate_features import encoder_xlmroberta

class Predictor:
    def __init__(self,weights_dir=pathlib.Path('modelweights'),model=segmenter(),encoder=encoder_xlmroberta,modelname=TOKENIZER):
        self.weights_dir = weights_dir / CHECKPOINTNAME
        self.encoder = encoder()
        self.model = model
        self.modelname = modelname

        print(10*"-")
        print("Loading model weights")
        self._load_segmenter_weights() #load weights
        print(10*"-"+'\n')

        print(10*"-")
        print("Initializing tokenizer")
        self._init_tokenizer() #cache tokenizer
        print(10*"-"+'\n')


    def _load_segmenter_weights(self):
        model = self.model
        weights_dir = self.weights_dir
        
        model.load_weights(weights_dir)
        
    
    def _split_mail(self,emailtext):
        self.emaillines =  emailtext.split("\n")
        
        assert len(self.emaillines)<=MAXLENMAIL, f'Can not classify mails longer than: {MAXLENMAIL}'
        
        return self.emaillines,len(self.emaillines)

    @functools.lru_cache
    def _init_tokenizer(self):
        modelname = self.modelname
        self.tokenizer = AutoTokenizer.from_pretrained(modelname)
        
        
    def _load_tokenizer(self,emaillines):
        tokenizer = self.tokenizer


        tokenized = tokenizer(emaillines,
                              padding=PAD,
                              truncation=True,
                              return_tensors='np',
                              )
        
        return tokenized
    
    
    def _extract_features(self,tokenized):
        encoder = self.encoder
        
        feature = encoder.predict([tokenized['input_ids'],tokenized['attention_mask']])
        feature = np.expand_dims(feature, axis=0)
        
        return feature     


    def _pad_feature(self,feature):
        feature_padded = tf.keras.preprocessing.sequence.pad_sequences(
            feature, maxlen=MAXTIMESTEP, dtype="float32", padding="post", value=MASKVALUE
            )
        return feature_padded
    

    def _classify_mail(self,feature):
        model = self.model
                
        
        pred = model.predict(feature,batch_size=16,verbose=1)
        
        return pred


    def predict(self,text_input):
        email = text_input
        emaillines,N_lines = self._split_mail(email)
        tokenized = self._load_tokenizer(emaillines)
        features = self._extract_features(tokenized)
        feature_padded = self._pad_feature(features)
        
        predicitons_proba = self._classify_mail(feature_padded) #before, load weights
        predicitons_proba = np.squeeze(predicitons_proba,axis=0)[0:N_lines,:]

        pred_lab_encoded = np.argmax(predicitons_proba,axis=1)
        pred_lab_decoded = np.array([LABELMAPPINGS_rev[i] for i in pred_lab_encoded])

        df_pred = pd.DataFrame({'text':self.emaillines,'predicitons':list(pred_lab_decoded)})

        return predicitons_proba,pred_lab_decoded,df_pred