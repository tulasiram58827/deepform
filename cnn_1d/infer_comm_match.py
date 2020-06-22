#!/usr/bin/env python3
#
# Runs inference on documents (files in a directory) referencing the
# full list of committees (or a specific section) from 2020.
# Currently not passing a basic test on 2020--best hypothesis is because
# the format is pretty different.
# Steps:
# load list of candidates
# load saved model
# load docs from test data directory
# score each doc against all the candidate committee names (currently specified by a type,
# trivial to extend to all types, though that's 11K candidates committees)

import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import sequence

import time

# TODO: these need to match the saved model properties
COMM_INPUT_LEN = 11
DOC_INPUT_LEN = 3000
INV_CHARS = {}

# utils
#-------------
def load_model(path='save test with pickle vocab'): #1dcnn_comm_match_model"):
  model = keras.models.load_model(path)
  return model

# TODO: replace "path" with directory of exported documents (each file is one string
# representing the doc)
def load_docs(path="docs_2020"):
  d = {}
  for f in os.listdir(path):
    f_contents = open(os.path.join(path, f), 'r').read()
    one_str = f_contents.replace("\n", " ")
    # TODO: this is because of current training setup--we probably want to embed a full vocab
    # and leave in capitalization as signal
    d[f] = one_str.lower()
  return d

# generate map of 2020 committees by type as defined here:
# https://www.fec.gov/campaign-finance-data/committee-type-code-descriptions/
# as of early June, here are some relevant counts:
# Presidential: type:  P  :  203
# Senate: type:  S  :  593
# House: type:  H  :  3138
def load_candidate_comms(csv_file="../../../../BigData/deepform/comm_summary_2020.csv"):
  f = open("COMMS_2020", 'w')
  df_cc = pd.read_csv(csv_file)
  comms = df_cc["CMTE_NM"].values
  comm_types = df_cc["CMTE_TP"].values
  comms_map = {}
  for c, t in zip(comms, comm_types):
    if t in comms_map:
      comms_map[t].append(c.lower())
    else:
      comms_map[t] = [c.lower()]
  return comms_map

# TODO this is copy-pasted from the training code and not particularly efficient
def make_old_block(docs, comms):
  # for each letter in each sliding window of the incoming document, append a column
  # consisting of the first comm_input_len letters of the committee name
  block = np.empty([len(docs), DOC_INPUT_LEN, COMM_INPUT_LEN + 1])
  # TODO: this would be more efficient in array syntax
  for i, d in enumerate(docs):
    comm = np.repeat(np.expand_dims(comms[i], axis=1), DOC_INPUT_LEN, axis=1)
    transpose_stack = np.transpose(np.vstack((d, comm)))
    block[i][:][:] = transpose_stack
  return block

# TODO: this is also copy-pasted, we want to fix vocabulary construction
def build_vocab(use_dict={}):
  tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
  # If we already have a character list, then replace the tk.word_index
  # If not, just skip below part

  # -----------------------Skip part start--------------------------
  # construct a new vocabulary
  if not use_dict:
    alphabet = " abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    for i, char in enumerate(alphabet):
      char_dict[char] = i + 1
  # Use char_dict to replace the tk.word_index
    tk.word_index = char_dict.copy()
  # Add 'UNK' to the vocabulary
    tk.word_index[tk.oov_token] = max(char_dict.values()) + 1

  # -----------------------Skip part end----------------------------
  else:
    tk.word_index = use_dict.copy()
    tk.word_index[tk.oov_token] = max(char_dict.values()) + 1
  return tk

# load everything
cmap = load_candidate_comms()
m = load_model()
docs = load_docs()

# try loading stored vocab previously used for training
char_dict = pickle.load(open("char_dict.pkl", 'rb'))
INV_CHARS = pickle.load(open("inv_chars.pkl", 'rb'))
tk = build_vocab(use_dict = char_dict)

# convert all docs to vocab
test_docs = tk.texts_to_sequences(docs.values())
test_doc_data = pad_sequences(test_docs, maxlen=DOC_INPUT_LEN, padding='post')
# convert all comms to vocab
test_comms = tk.texts_to_sequences(cmap["P"])
test_comms_data = pad_sequences(test_comms, maxlen=COMM_INPUT_LEN, padding='post')
print("TEST: ", test_comms_data)

np_docs = np.array(test_doc_data, dtype='float32')
np_comms = np.array(test_comms_data, dtype='float32')

# repeat each doc * np_comms times
for doc in np_docs:
  test_block = make_old_block([doc for i in range(len(np_comms))], np_comms)
  preds = m.predict(test_block)
  #print("PREDS: ", preds)
  argmax = np.argmax([p[0] for p in preds])
  print("NAME: ", cmap["P"][argmax])

#print(cmap["P"])
