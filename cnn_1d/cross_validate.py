#!/usr/bin/env python3

# Trains a simple 1D CNN architecture to predict whether a given document
# includes a given committee name as the correct payer (binary classification:
# the model literally returns a score < 0.5 if False and > 0.5 if True)
# Intended to be sweep-compatible. Run with "-h" to see all the options

import argparse

import time

import tensorflow as tf 

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Input, Reshape, Embedding, Flatten
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, AveragePooling1D, MaxPooling1D

import numpy as np
import os
import pandas as pd
import pickle
import wandb
from wandb.keras import WandbCallback
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

RACES = ["Presidential", "US Senate", "US House", "State", "Local"]
COMM_CODES = ["P", "S", "H"]

# cross validation scenario:
# train model on 900, test on 100

char_dict = {}
INV_CHARS = {}

# modify to your project if you like
# note that if you run a sweep, you will need to set these as environment variables as follows
# in order for them to work, or add them to the sweep.yaml file
# export WANDB_ENTITY=
# export WANDB_PROJECT=
WB_PROJECT_NAME = "cnn_1d"
WB_ENTITY = "deepform"

# Settings / Hyperparameters
# can be modified here, in the command-line, or through a sweep config
MODEL_NAME = ""
DATA_PATH = ""

# full 2012 data contains 9018 documents
NUM_TRAIN = 900 # max is 7218
NUM_VAL = 0
NUM_TEST = 100 # max is 1800
EPOCHS = 1

# number of distractors (non-matching committee names) to choose from
NUM_CLASSES = 420 #600
BATCH_SIZE = 64
doc_input_len = 3000

# observed stats on 2012 data from 9018 labels:
# average committee name label length is 22, max is 80
comm_input_len = 15 #11

# network architecture 
filters = 300
hidden_dims = 600
kernel_size = 6
dropout_1 = 0.2
dropout_2 = 0.4
optimizer = "SGD"
learning_rate = 0.002
distractors = 10

# log receipt text as a string to wandb.Table
# by default, logs the most confidence false positives as these will be the
# most problematic
class LogCallback(WandbCallback):
  
  def on_epoch_end(self, epoch, epoch_logs, false_pos=True):
    # limit number of entries in the table
    num_log = 20
  
    table = wandb.Table(columns = ["Document", "Committee Label", "Score", "Truth"]) 
    # TODO: runs prediction on full validation data, we may not want this
    preds = self.model.predict(self.validation_data[0])  
    truth = self.validation_data[1] 

    if false_pos:
      # sort predictions with highest-scoring false positives on top
      pairs = zip(preds, truth)
      fp = [[i, p[0]] for i, p in enumerate(pairs) if p[0] > 0.5 and p[1] == 0]
      print("false pos count: ", len(fp), " out of ", len(self.validation_data[0])) 
      fp.sort(key=lambda x: x[1], reverse=True)
      top_fp = fp[:num_log]
      print("top fp: ", top_fp)
    
      for doc_id, score in top_fp:
        doc = self.validation_data[0][doc_id]
        try:
          dtext = "".join([INV_CHARS[int(row[0])] for row in doc[:200]])
          ctext = "".join([INV_CHARS[int(i_c)] for i_c in doc[0][1:]])
        except:
          dtext = "N/A"
          ctext = "N/A"
        table.add_data(dtext, ctext, score, "0")
      wandb.log({"top false positives" : table})
    else:
      # regular predictions
      for doc, pred, truth in zip(self.validation_data[0][:num_log], preds, self.validation_data[1][:num_log]):
        dtext = "".join([INV_CHARS[int(row[0])] for row in doc[:200]])
        ctext = "".join([INV_CHARS[int(i_c)] for i_c in doc[0][1:]])
        table.add_data(dtext, ctext, pred[0], truth)  
     
      wandb.log({"predictions" : table})

# utils
#--------------------------------
def load_class_from_module(module_name):
  components = module_name.split('.')
  mod = __import__(components[0])
  for comp in components[1:]:
    mod = getattr(mod, comp)
  return mod

def load_optimizer(optimizer, learning_rate):
  """ Dynamically load relevant optimizer """
  optimizer_path = "tensorflow.keras.optimizers." + optimizer
  optimizer_module = load_class_from_module(optimizer_path)
  return optimizer_module(lr=learning_rate)

def cd_stack(doc, comm, doc_len, p=False):
  comm_block= np.repeat(np.expand_dims(comm, axis=1), doc_len, axis=1)
  if p:
    print("DOC: ", doc.shape)
    print("COMM: ", comm.shape)
    print("BLOCK: ", comm_block.shape)
  return np.transpose(np.vstack((doc, comm_block)))

def make_block(docs, comms, cfg):
  # for each letter in each sliding window of the incoming document, append a column
  # consisting of the first comm_input_len letters of the committee name
  block = np.empty([docs.shape[0], cfg.doc_input_len, cfg.comm_input_len + 1])
  # TODO: this would be more efficient in array syntax
  for i, d in enumerate(docs): 
    block[i][:][:] = cd_stack(d, comms[i], cfg.doc_input_len)
  return block 

# load training data and encode into 1-hot tensors
# shamelessly copy-pastad from different script
def BuildInputTensor(cfg):

   # train = "../../../../BigData/deepform/training.csv"
    train = "../../../../BigData/deepform/fcc-data-2020-sample-updated.csv"
    filings = "../../../../BigData/deepform/ftf-all-filings.tsv"

    truncate_length = cfg.doc_input_len
    num_samples = cfg.num_train + cfg.num_val

    dft = pd.read_csv(train)  # , nrows = docs)
    dff = pd.read_csv(filings, sep='\t')

    df_all = pd.merge(left=dft, right=dff, how='left',
                      left_on='slug', right_on='dc_slug')
    #df_all = df_all[['slug', 'page', 'x0', 'y0', 'x1', 'y1', 'token', 'gross_amount_x', 'committee']]
    df_all = df_all[['slug', 'token', 'committee']]

    df_group = df_all.groupby(['slug', 'committee'])['token'].apply(
        lambda a: ' '.join([str(x) for x in a])).reset_index()
    print(df_group.shape)
    print('number of documents')

    df_group['text'] = df_group['token'].str.slice(0, truncate_length)
    df_group['committee'] = '\t' + df_group['committee'] + '\n'
    #df_group['committee'] = str(df_group['committee']).strip().lower()

    df_group.drop(['token'], axis=1)

    print(df_group['committee'][:3])

    target_texts = df_group['committee'][:num_samples]
    input_texts = df_group['text'][:num_samples]
    return input_texts, target_texts

def read2020(cfg=None):
  train = "../../../../BigData/deepform/fcc-data-2020-sample-updated.csv"
  dft = pd.read_csv(train)
  dft = dft[~dft["advertiser"].isnull()]
  x = dft["text"].values
  y = dft["advertiser"].values
  office = dft["office_type"].values
  # clean up the nans
  clean_x = [s.replace("\n", " ") for s in x]
  return clean_x, y, office

def load_candidate_comms(comm_len, csv_file="../../../../BigData/deepform/comm_summary_2020.csv"):
  f = open("COMMS_2020", 'w')
  df_cc = pd.read_csv(csv_file)
  comms = df_cc["CMTE_NM"].values
  comm_types = df_cc["CMTE_TP"].values
  comms_map = {}
  for c, t in zip(comms, comm_types):
    if t in comms_map:
      comms_map[t].append(c.lower()[:comm_len]) 
    else:
      comms_map[t] = [c.lower()[:comm_len]]
  return comms_map

def build_comm_map(cfg, x, y):
# build committee dictionary
# TOOD: is this necessary? pickle it
  labels = {}
  for comm_name in y:
    try:
      clean_name = comm_name.strip().lower()[:cfg.comm_input_len]
    except:
      print("WTF: ", comm_name)
      continue
    if clean_name in labels:
      labels[clean_name] += 1
    else:
      labels[clean_name] = 1
  label_names = list(labels.keys())
  print("num labels: ", len(label_names))

  label_ints = { l : _id for _id, l in enumerate(label_names)}
  label_ints["UNK"] = len(label_names)
  #print("labels: ", label_ints)
  print("label length: ", len(label_ints))
  
  # sorted dictionary
  l_sort = sorted(labels.items(), key=lambda kv: kv[1], reverse=True)
  print("TOP 10 LABELS: ", l_sort[:10])

  id_to_label = { v : k for k, v in label_ints.items() }
  return label_ints, id_to_label

# reshuffle so that each correct label has D=1 distractors
def pair_with_distractors(cfg, x, y, label_ints, id_to_label, D=1):
  x_doc = []
  x_comm = []
  new_y = []
  for x_sample, y_label in zip(x, y):
    # append correct comm
    clean_label = y_label.strip().lower()[:cfg.comm_input_len]
    x_doc.append(x_sample)
    x_comm.append(clean_label)
    real_label_id = label_ints[clean_label]
    new_y.append(1)
    # if no distractors
    if D == 0:
      continue
    # append D incorrrect comms
    false_labels = np.random.choice(cfg.num_classes, cfg.distractors, replace=False)
    for f_l in false_labels:
      # make sure that we don't shoot ourselves in the foot by appending 
      # the real comm name with a false label
      if f_l != real_label_id:
        x_doc.append(x_sample)
        x_comm.append(id_to_label[f_l])
        new_y.append(0)
  return x_doc, x_comm, new_y

# TODO: bring back embedding/1 hot encoding/to_categorical
#from keras.utils import to_categorical
#train_data_block = to_categorical(train_data_block)
#test_data_block = to_categorigcal(test_data_block)
#y_verdict = to_categorical(y_verdict, 2)
#test_labels = to_categorical(test_labels, 2)

# this is a basic 1D CNN
def vanilla_1dcnn(cfg):
  model = Sequential()

  # TODO: consider bringing back an embedding layer, this is hard with
  # 2D input (would need to flatten then reshape)
  # we start off with an efficient embedding layer which maps
  # our vocab indices into embedding_dims dimensions
  # ["The committee on America..."]
  # [V["T"], V["h"], V["e"], V[" "] ...]
  #  model.add(Embedding(cfg.num_features,
  #                    cfg.embed_dims,
  #                    input_length=(3000, 11)))
  #  model.add(Dropout(cfg.drop_1))
  
  # we add a Convolution0D, which will learn filters
  # word group filters of size filter_length:
  model.add(Conv1D(cfg.filters,
                 cfg.kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
  model.add(MaxPooling1D(pool_size=2))
  model.add(Dropout(cfg.dropout_1))
 
  # TODO: could stack these layers 
  #model.add(Conv1D(100,
  #               5,
  #               padding='valid',
  #               activation='relu',
  #               strides=1))
  #model.add(MaxPooling1D(pool_size=2))
  #model.add(Dropout(cfg.dropout_1))
  
  model.add(Flatten())
  # We add a vanilla hidden layer:
  model.add(Dense(cfg.hidden_dims, activation="relu"))
  model.add(Dropout(cfg.dropout_2))

  # this secondary layer doesn't help much
  #model.add(Dense(100))
  #model.add(Dropout(0.5))
  #model.add(Activation('relu'))

  # TODO: how do we adjust this cutoff if we like?
  model.add(Dense(1, activation="sigmoid"))

  lr_optimizer = load_optimizer(cfg.optimizer, cfg.learning_rate)
  model.compile(loss='binary_crossentropy',
              optimizer=lr_optimizer,
              metrics=['accuracy'])
  return model

def train_cnn(args):

  config = {
    "num_train" : args.num_train,
    "num_val" : args.num_val,
    "num_classes" : NUM_CLASSES,
    "epochs" : args.epochs,
    # hardcoded for now--change if logging a substantially different model type
    "model" : "cross_val",
    "batch_size" : args.batch_size,
    "doc_input_len" : doc_input_len,
    "comm_input_len" : args.comm_input_len,
    "filters" : args.filters,
    "kernel_size" : args.kernel_size,
    "hidden_dims" : args.hidden_dims,
    "dropout_1" : args.dropout_1,
    "dropout_2" : args.dropout_2,
    "learning_rate" : args.learning_rate,
    "optimizer" : args.optimizer,
    "distractors" : args.distractors
  }
   
  # if a special model name is not set from the command line, 
  # compose model name from relevant hyperparameters
  run_name = args.model_name
  if not run_name:
    run_name = "ip c" + str(config["comm_input_len"]) + " f " + str(config["filters"]) + " ds " + str(config["distractors"]) + \
               " hd " + str(config["hidden_dims"]) + " d1 " + str(config["dropout_1"]) + " lr" + str(config["learning_rate"])

  print("this run is called: ", run_name)
  wandb.init(project=WB_PROJECT_NAME, name=run_name, entity=WB_ENTITY)
  cfg = wandb.config
  cfg.setdefaults(config)

  # load training data
  x, y, office = read2020(cfg) #BuildInputTensor(cfg)
  # or read from pre-saved file
  #docs_file = open("DOCS_9K", 'r')
  #comms_file = open("COMMS_9K", 'r')
  #x = docs_file.readlines()
  #y = comms_file.readlines()
  print("Number of documents: ", len(x))
  print("Number of committees: ", len(y))
  #print(y
  #)
  #comms = load_candidate_map
  label_ints, id_to_label = build_comm_map(cfg, x, y)
  # parse training data (TODO: factor out, fix train/val split to reduce variance)
  all_data = list(zip(x, y, office))
  np.random.shuffle(all_data)

  x, y, office = list(zip(*all_data))
  # validation mode
  if cfg.num_val: 
    train_data = x[:cfg.num_train]
    val_data = x[cfg.num_train:cfg.num_train + cfg.num_val]
    train_labels = y[:cfg.num_train]
    val_labels = y[cfg.num_train:cfg.num_train + cfg.num_val]
    train_and_val(cfg, train_data, train_labels, val_data, val_labels, label_ints, id_to_label)
  else:
    print("ENTERING CROSS VALIDATION")
    # test mode, 10 times!
    B = 100
    for i in range(0, len(x) - B, B):
      test_data = x[i:i+B]
      test_labels = y[i:i+B]
      train_data = x[i+B:]
      train_labels = y[i+B:]
      cross_val(cfg, train_data, train_labels, test_data, test_labels, label_ints, id_to_label, office)

def cross_val(cfg, train_data, train_labels, test_data, test_labels, label_ints, id_to_label, office):
  st_t = time.time()
  x_doc, x_comm, y_verdict = pair_with_distractors(cfg, train_data, train_labels, label_ints, id_to_label)
  y_doc, y_comm, test_verdict = pair_with_distractors(cfg, test_data, test_labels, label_ints, id_to_label, D=0) 
  comms_map = load_candidate_comms(cfg.comm_input_len)
  cb = {}
  

  # we could in fact make fixed test blocks: pres, sen, house...

  # TODO: this tokenization is copy-pastad from a Keras 1D example--probably worth double-checking
  # =======================Convert string to index================
  # Tokenizer
  tk = Tokenizer(num_words=None, char_level=True, oov_token='@')
  tk.fit_on_texts(train_data)
  # If we already have a character list, then replace the tk.word_index
  # If not, just skip below part

  # -----------------------Skip part start--------------------------
  # construct a new vocabulary
  #alphabet = " abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
  #for i, char in enumerate(alphabet):
  #  char_dict[char] = i + 1
  #  INV_CHARS[i+1] = char
  #INV_CHARS[70] = "~"
  #INV_CHARS[0] ="@"
 
  # Use char_dict to replace the tk.word_index
  #tk.word_index = char_dict.copy()
  # Add 'UNK' to the vocabulary
  #tk.word_index[tk.oov_token] = max(char_dict.values()) + 1
 # pickle both vocabs for reloading
  #pickle.dump(INV_CHARS, open("inv_chars.pkl", "wb"))
  #print("WORD index: ", tk.word_index)
  pickle.dump(tk.word_index, open("tk_word_index.pkl", "wb"))
  INV_CHARS = {v : k for k, v in tk.word_index.items()}
 
  # -----------------------Skip part end----------------------------

  # Convert string to index
  train_sequences = tk.texts_to_sequences(x_doc)
  train_comms = tk.texts_to_sequences(x_comm)
  test_texts = tk.texts_to_sequences(y_doc)
  test_comms = tk.texts_to_sequences(y_comm)

 
  # Padding
  train_doc_data = pad_sequences(train_sequences, maxlen=cfg.doc_input_len, padding='post')
  train_comm_data = pad_sequences(train_comms, maxlen=cfg.comm_input_len, padding='post')
  test_data = pad_sequences(test_texts, maxlen=cfg.doc_input_len, padding='post')
  #test_comm_data = pad_sequences(test_comms, maxlen=cfg.comm_input_len, padding='post')

  # Convert to numpy arrays
  train_data = np.array(train_doc_data, dtype='float32')
  comm_data = np.array(train_comm_data, dtype='float32')
  test_data = np.array(test_data, dtype='float32')
  #test_comm_data = np.array(test_comm_data, dtype='float32')
  #test_labels = np.array(test_labels, dtype='float32')
  y_verdict = np.array(y_verdict, dtype='float32')

  # same for target comms
  cb["P"] = np.array(pad_sequences(tk.texts_to_sequences(comms_map["P"]), maxlen=cfg.comm_input_len, padding='post'))
  cb["S"] = np.array(pad_sequences(tk.texts_to_sequences(comms_map["S"]), maxlen=cfg.comm_input_len, padding='post'))
  cb["H"] = np.array(pad_sequences(tk.texts_to_sequences(comms_map["H"]), maxlen=cfg.comm_input_len, padding='post'))


  train_data_block = make_block(train_data, comm_data, cfg)
  #test_data_block = make_block(test_data, test_comm_data, cfg)
  print("TRAIN DATA BLOCK: ", train_data_block.shape)
  #print("TEST DATA BLOCK: ", test_data_block.shape)

  print("TRAIN START: ", time.time()-st_t)

  model = vanilla_1dcnn(cfg)
  model.fit(train_data_block, y_verdict,
          batch_size=cfg.batch_size,
          epochs=cfg.epochs,
          callbacks=[WandbCallback(save_model=True)]) #, LogCallback()])
 
  print(model.summary())
  print("FINISHED TRAINING", time.time()-st_t)
  # now inference mode
  # we need to pair every test doc with every comm,
  # then loop over all the candidate comms
  start_t = time.time()
  print("TYPE: ", type(office))
  small = office[0:10]
  print("SMALL: ", small)

  #test_block = np.empty([5, cfg.doc_input_len, cfg.comm_input_len +1])
  for i, s in enumerate(small):
    curr_office = office_code(office[i])
    print("curr_office: ", curr_office)
    if not curr_office:
      continue
    else:
      office_candidates = cb[curr_office]
      test_pairs = np.empty([len(office_candidates), cfg.doc_input_len, cfg.comm_input_len + 1])      
      for o in office_candidates:
        #print("O: ", o) 
        test_pairs[i][:][:] = (cd_stack(test_data[i], o, cfg.doc_input_len)) #, p=True))
      scores = model.predict(test_pairs) #[test_pairs[0]])
      print("PREDS: ", scores)
      print(comms_map[curr_office])
      print("TIME SO FAR: ", time.time()-start_t) 

def office_code(long_code):
  if long_code == "Presidential":
    return "P"
  elif long_code == "US Senate":
    return "S"
  elif long_code == "US House":
    return "H"
  else:
    print("office: ", long_code, ": too many to try")
    return ""
   


 
def train_and_val(cfg, train_data, train_labels, val_data, val_labels, label_ints, id_to_label):
  x_doc, x_comm, y_verdict = pair_with_distractors(cfg, train_data, train_labels, label_ints, id_to_label)
  y_doc, y_comm, test_labels = pair_with_distractors(cfg, val_data, val_labels, label_ints, id_to_label)

  # TODO: this tokenization is copy-pastad from a Keras 1D example--probably worth double-checking
  # =======================Convert string to index================
  # Tokenizer
  tk = Tokenizer(num_words=None, char_level=True, oov_token='@')
  tk.fit_on_texts(train_data)
  # If we already have a character list, then replace the tk.word_index
  # If not, just skip below part

  # -----------------------Skip part start--------------------------
  # construct a new vocabulary
  #alphabet = " abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
  #for i, char in enumerate(alphabet):
  #  char_dict[char] = i + 1
  #  INV_CHARS[i+1] = char
  #INV_CHARS[70] = "~"
  #INV_CHARS[0] ="@"
 
  # Use char_dict to replace the tk.word_index
  #tk.word_index = char_dict.copy()
  # Add 'UNK' to the vocabulary
  #tk.word_index[tk.oov_token] = max(char_dict.values()) + 1
 # pickle both vocabs for reloading
  #pickle.dump(INV_CHARS, open("inv_chars.pkl", "wb"))
  print("WORD index: ", tk.word_index)
  pickle.dump(tk.word_index, open("tk_word_index.pkl", "wb"))
  INV_CHARS = {v : k for k, v in tk.word_index.items()}
 
  # -----------------------Skip part end----------------------------

  # Convert string to index
  train_sequences = tk.texts_to_sequences(x_doc)
  train_comms = tk.texts_to_sequences(x_comm)
  test_texts = tk.texts_to_sequences(y_doc)
  test_comms = tk.texts_to_sequences(y_comm)

  # Padding
  train_doc_data = pad_sequences(train_sequences, maxlen=cfg.doc_input_len, padding='post')
  train_comm_data = pad_sequences(train_comms, maxlen=cfg.comm_input_len, padding='post')
  test_data = pad_sequences(test_texts, maxlen=cfg.doc_input_len, padding='post')
  test_comm_data = pad_sequences(test_comms, maxlen=cfg.comm_input_len, padding='post')

  # Convert to numpy arrays
  train_data = np.array(train_doc_data, dtype='float32')
  comm_data = np.array(train_comm_data, dtype='float32')
  test_data = np.array(test_data, dtype='float32')
  test_comm_data = np.array(test_comm_data, dtype='float32')
  test_labels = np.array(test_labels, dtype='float32')
  y_verdict = np.array(y_verdict, dtype='float32')

  train_data_block = make_block(train_data, comm_data, cfg)
  test_data_block = make_block(test_data, test_comm_data, cfg)
  print("TRAIN DATA BLOCK: ", train_data_block.shape)
  print("TEST DATA BLOCK: ", test_data_block.shape)

  model = vanilla_1dcnn(cfg)
  model.fit(train_data_block, y_verdict,
          batch_size=cfg.batch_size,
          epochs=cfg.epochs,
          validation_data=(test_data_block, test_labels),
          callbacks=[WandbCallback(save_model=True), LogCallback()])
  
  # will default to saving the model by the run name 
  model.save(wandb.run.name)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
    "-m",
    "--model_name",
    type=str,
    default=MODEL_NAME,
    help="Name of this model/run (model will be saved to this file)")
  parser.add_argument(
    "-d",
    "--data_path",
    type=str,
    default=DATA_PATH,
    help="Path to data, containing train/ and val/")
  parser.add_argument(
    "-nt",
    "--num_train",
    type=int,
    default=NUM_TRAIN, 
    help="Total number of training examples to use")
  parser.add_argument(
    "-nv",
    "--num_val",
    type=int,
    default=NUM_VAL, 
    help="Total number of validation examples to use")
  parser.add_argument(
    "-ns",
    "--num_test",
    type=int,
    default=NUM_TEST,
    help="Total number of test examples to use")
  parser.add_argument(
    "-b",
    "--batch_size",
    type=int,
    default=BATCH_SIZE,
    help="Number of images in training batch")
  parser.add_argument(
    "-e",
    "--epochs",
    type=int,
    default=EPOCHS,
    help="Number of training epochs")
  parser.add_argument(
    "--comm_input_len",
    type=int,
    default=comm_input_len,
    help="Max committee name input length")
  parser.add_argument(
    "--filters", 
    type=int,
    default=filters,
    help="number of filters to learn")
  parser.add_argument(
    "--kernel_size", 
    type=int,
    default=kernel_size,
    help="number of characters to consider in sliding window")
  parser.add_argument(
    "--hidden_dims", 
    type=int,
    default=hidden_dims,
    help="number of filters to learn")
  parser.add_argument(
    "--dropout_1", 
    type=float,
    default=dropout_1,
    help="dropout 1")
  parser.add_argument(
    "--dropout_2", 
    type=float,
    default=dropout_2,
    help="dropout 2")
  parser.add_argument(
    "--learning_rate", 
    type=float,
    default=learning_rate,
    help="learning rate") 
  parser.add_argument(
    "--optimizer", 
    type=str,
    default=optimizer,
    help="optimizer")
  parser.add_argument(
    "--distractors", 
    type=int,
    default=distractors,
    help="number of distractors")
  parser.add_argument(
    "-q",
    "--dry_run",
    action="store_true",
    help="Dry run (do not log to wandb)")
  args = parser.parse_args()

  # easier testing--don't log to wandb if dry run is set
  if args.dry_run:
    os.environ['WANDB_MODE'] = 'dryrun'

  train_cnn(args) 
