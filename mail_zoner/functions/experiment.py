import pathlib
# import numpy as np
import time

import tensorflow as tf
import matplotlib.pyplot as plt

from ..functions.data_generator import DatasetGenerator
from ..parameters import EPOCHS,BATCHSIZE,CHECKPOINTNAME,NLABELS
# from ..util.util import compute_class_weights

def plot_history(hist):
    def plot_result(hist,item):
        plt.plot(hist.history[item], label=item)
        plt.plot(hist.history["val_" + item], label="val_" + item)
        plt.xlabel("Epochs")
        plt.ylabel(item)
        plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
        plt.legend()
        plt.grid()
        plt.show()

    plot_result(hist,"loss")
    plot_result(hist,"categorical_accuracy")
    plot_result(hist,"precision")
    plot_result(hist,"recall")
    plot_result(hist,"prc")


class Trainer:
    def __init__(self,model,data_path_train,data_path_val,data_path_test,log_path_train,log_path_evaluation,model_path_checkpoint):
        time_now = str(time.time()).split(".")[0]
        self.model = model
        
        self.data_path_train = data_path_train
        self.data_path_val = data_path_val
        self.data_path_test = data_path_test
        
        self.log_path_train = log_path_train / time_now
        self.log_path_evaluation = log_path_evaluation / time_now
        self.model_path_checkpoint = model_path_checkpoint / time_now

        # Init generators
        self.init_generators()

        # Compute classweights
        # self.init_classweights()

        
    def init_generators(self):
        data_path_train = self.data_path_train
        data_path_val = self.data_path_val
        data_path_test = self.data_path_test
        
        self.datagen_train = DatasetGenerator(data_path=data_path_train,batch_size=BATCHSIZE,shuffle=True)
        self.datagen_val = DatasetGenerator(data_path=data_path_val,batch_size=BATCHSIZE,shuffle=True)
        self.datagen_test = DatasetGenerator(data_path=data_path_test,batch_size=BATCHSIZE,shuffle=True)


    # def init_classweights(self):
    #     data_path_train = self.data_path_train
    #     data_path_glob = data_path_train.glob('**/*.npy') #complete path to all samples (including labels _l.npy)

    #     elements = sorted([i.stem for i in data_path_glob])
    #     ids_list = list(set([i.split("_")[0] for i in elements]))

    #     labels_tot = []
    #     for key in ids_list:
    #         key_label = f"{key}_l"

    #         label_dir = data_path_train.joinpath(f"{key_label}.npy")
    #         label_onehot = np.load(label_dir)
    #         label_encoded = list(np.argmax(label_onehot,axis=1))

    #         labels_tot.extend(label_encoded)

    #     # Calculate cw
    #     self.cw = {key:1.0 for key in range(NLABELS)}
    #     cw_temp = compute_class_weights(labels_tot) #when not all classes are in set, not all keys are present
    #     for key,val in cw_temp.items():
    #         self.cw[key] = val
        

    def config_tensorboard(self,logdir):
        return tf.keras.callbacks.TensorBoard(log_dir=logdir,write_graph=False)


    def train(self): #to run tensorboard: tensorboard --logdir <path_to_train_logs= i.e. data_webis/logs_train>
        model = self.model
        datagen_train = self.datagen_train
        datagen_val = self.datagen_val
        # cw = self.cw
        
        log_path_train = self.log_path_train
        model_path_checkpoint = self.model_path_checkpoint / CHECKPOINTNAME
        
        modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(model_path_checkpoint,
                                                             monitor="val_loss",
                                                             verbose=0,
                                                             save_best_only=True,
                                                             save_weights_only=True,
                                                             mode="min")

        
        history = model.fit(x=datagen_train,
                            epochs=EPOCHS,
                            callbacks=[self.config_tensorboard(log_path_train),modelcheckpoint],
                            validation_data=datagen_val,
                            # class_weight=cw,
                            workers=1,
                            use_multiprocessing=False, #enable this if you work on a GPU and have many workers
                            max_queue_size=10)
        return history

   
    def evaluate(self): #to run tensorboard: tensorboard --logdir data/logs_evaluation
        model = self.model
        log_path_evaluation = self.log_path_evaluation
        datagen_test = self.datagen_test
        
        
        results = model.evaluate(x=datagen_test,
                                 max_queue_size=10,
                                 callbacks=[self.config_tensorboard(log_path_evaluation)],
                                 workers=1,
                                 use_multiprocessing=False,
                                 return_dict=True)
        return results