# Introduction 
The mail-Zoner project aims to automatically clean noise from sequencial dependent structured text, which for example is characteristical for e-mails. An e-mail typically starts with a salutation, followed by the actual information to transmit and some closing farewell message. One can aim to segment mails into zones, where the goal would be to extract the actual information and discard the noise (e.g. salutations, farewell message, thank you message).

In this project, raw text from e-mail were split by newline character, where the number of lines were defined as the length of an e-mail. The supervised approach was to classify each of these lines to the dedicated labels.

## Model
The architecture consists of an encoder and a segmenter part, where the encoder generates features and the segmenter classifies lines. It has multilingual capabilities and considers sequential nature of input with LSTM.

###### Encoder: XLM-RoBERTa
To leverage cross lingual ability with state-of-art understanding of text, [XLM-RoBERTa](https://arxiv.org/abs/1911.02116) architecture was chosen as the encoder. Note that the weights of the encoder were freezed (transfer learning). The last four latent states were concatenated and then averaged to obtain the feature vector of each line of an email.

###### Segmenter: Bi-LSTM-Dense
Because of the sequential dependency of each line (i.e. the text has some structure, which we must take into account), a Bi-LSTM-Dense architecture was chosen. Note that one could also use a [BiLSTM-CRF](https://arxiv.org/abs/1508.01991) architecture to further tune the model, if needed.

## Experiment with public dataset: [Webis Gmane Email Corpus 2019](https://zenodo.org/record/3766985)
Data for the first experiment is in `data_webis\annotations-final-train.jsonl` and `data_webis\annotations-final-validation.jsonl`. It has been download from this [repository](github.com/webis-de/acl20-crawling-mailing-lists/tree/master/annotations).

###### Training procedure
Training with the public dataset was done in Google Colab on a Tesla P100 GPU, having ~2h for feature extraction and ~30min for the segmentation part. The encoder saves features in the folder `data_webis\features`, where in the experiment it was fed with a data generator to the segmenter. Training logs were saved in `data_webis\logs_train\{timestamp}`, and the evaluation logs in `data_webis\logs_evaluation \{timestamp}`. The model was checkpointed in the folder `data_webis\modelcheckpoint\{timestamp}`. This means that the model weights, which were trained, have been saved there and only the best model has been checkpointed.

# Getting Started

## 1. Installing packages
###### Poetry: Poetry has problems installing matplotlib and tensorflow
- Install poetry https://python-poetry.org/docs/#installation
- Run poetry shell to activate your environment
- Run poetry install to install dependencies
###### Pip
- Run python -m pip install -U tensorflow
- Run python -m pip install -U matplotlib

## 2. Retraining the model
###### Feature generation & storage
- Run `ml_classification_preprocessing_email_zone\notebooks\generate_features.ipynb` to extract and save features to `data_webis\features`

###### Starting the experiment
- Run `ml_classification_preprocessing_email_zone\notebooks\experiment.ipynb` to train the segmenter. The model is checkpointed (best model weights are stored) in `data_webis\modelcheckpoint\{timestamp}` and the training and evaluation logs are saved in `data_webis\logs_train\{timestamp}` and `data_webis\logs_evaluation \{timestamp}`

###### View the results
- One can simply run `tensorboard --logdir <path_to_train_logs> i.e. tensorboard --logdir data_webis/logs_train/123456` to see the training graphs

## 3. Making a prediction
- Run `ml_classification_preprocessing_email_zone\notebooks\predict.ipynb` with the approriate inputs the make a prediction

# Contribute
- Connect to feature store for generated features (temporary directory to store features is now `data_webis\features` )
- Add class weights or sample weights to tackle imbalanced dataset
- TensorFlow serving
    - Add tensorflow serving to predict.py
    - Figure a way to cache huggingface transformer model s.t. it is faster when predicting
- Kubeflow integration
    - TensorBoard
    - Add confusion matrix `tf.math.confusion_matrix` to evaluate model performance for each class when evaluating
- Optional: Change dense layer with CRF (conditional random field) in segmenter