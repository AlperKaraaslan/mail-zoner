import tensorflow as tf

from ..parameters import NLABELS,MAXTIMESTEP,MASKVALUE

def compile_segmenter(model):
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

    metrics=[tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall'),
             tf.keras.metrics.AUC(name='prc', curve='PR')
             ]

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=metrics)
    
def segmenter():
    features = tf.keras.layers.Input(shape=(MAXTIMESTEP,3072,), dtype=tf.float32)

    masking = tf.keras.layers.Masking(mask_value=MASKVALUE)(features)

    lstm = tf.keras.layers.LSTM(units=64,return_sequences=True)
        
    bilstm = tf.keras.layers.Bidirectional(lstm)(masking)
    
    dropout = tf.keras.layers.Dropout(0.25)(bilstm)

    dense = tf.keras.layers.Dense(NLABELS,activation='softmax')

    pred =  tf.keras.layers.TimeDistributed(dense)(dropout)
    
    model = tf.keras.Model(inputs=features,outputs=pred)

    compile_segmenter(model)

    return model