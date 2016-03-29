# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 17:22:24 2016

@author: oafolabi
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import string
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn import rnn_cell

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train')
    parser.add_argument('--sample')
    parser.add_argument('--cell',choices=['vanilla','gru', 'lstm'],default='lstm')
    parser.add_argument('--num_layers',type=int,default=2)
    parser.add_argument('--rnn_size',type=int,default=256)
    parser.add_argument('--seq_length',type=int,default=40)
    
    args = parser.parse_args()
    argv = vars(args)    
    
    for key in argv:
        print(key)
        print(argv[key])
    process_data(argv['train'])
    
c2i,i2c = {},{}    
def process_data(file_name):
    
    with open(file_name) as f:
         fstring = ""
         for line in f:
             for char in line:
                 it_prints=char in string.printable #filter non-printable characters
                 if not char in c2i and it_prints: #if no index exist for character
                    #add to dictionaries
                    c2i[char] = len(c2i) 
                    i2c[len(c2i)-1] = char
             fstring = fstring + line
         #print(fstring)
         print(c2i)
         print(i2c)
         print(mini_batcher(fstring,32,12))

#mini batcher:
#   Creates a batch of sequences to train on
#Input:
#   src_str: dataset to pull from
#   bat_size: number of sequences in batch
#   seq_len: length of each sequence
#
#Output:
#   list of seuqnces, each seqence is a list of indices      
def mini_batcher(src_str,bat_size,seq_len):
    batch = {}
    for i in range(bat_size):
        st_ind = np.random.randint(len(src_str)-seq_len)
        batch[i]=[c2i[char] for char in src_str[st_ind:st_ind+seq_len]]
    return batch  
    
def split_inputs(inputs,length):
	split_inputs = tf.split(0,length,inputs)
	return [tf.squeeze(input_,[0]) for input_ in split_inputs]


#model
#   Takes inputs from the command line and builds the model for the neural network
#returns:
#   inp_ph: Input placeholder. Structure for inputting data into the neural net
#   top_hids: Top layer hidden states of each cell
#   fin_hids: All hidden layers of final cell         
def model( opt ):
    if opt['cell'] == 'vanilla':
        cell = rnn_cell.BasicRNNCell(opt['rnn_size'])
        cell0 = rnn_cell.BasicRNNCell(opt['rnn_size'],input_size=len(c2i)+1)
    elif opt['cell'] == 'gru':
        cell = rnn_cell.GRUCell(opt['rnn_size'])
        cell0 = rnn_cell.GRUCell(opt['rnn_size'],input_size=len(c2i)+1)
    elif opt['cell'] == 'lstm':
        cell = rnn_cell.BasicLSTMCell(opt['rnn_size'])
        cell0 = rnn_cell.BasicLSTMCell(opt['rnn_size'],input_size=len(c2i)+1)
        
    #build input placeholder
    inp_ph = tf.placeholder(tf.float32,[opt['seq_length'],None,len(c2i)+1])
    inputs = split_inputs(inp_ph,opt['seq_length'])      
        
    multicell = rnn_cell.MultiRNNCell([cell0]+[cell]*(opt['num_layers']-1)) #stack copies of cells to proper number of layers
    multicell_out = rnn_cell.OutputProjectionWrapper(multicell,len(c2i)+1) #project onto correct dimensionality
    top_hids, fin_hids = rnn.rnn(multicell_out,inputs,dtype=tf.float32) #repeat multicell for sequence len
    
    return inp_ph,top_hids,fin_hids  
    
    
    
def add_training_nodes():

   return 
    
def train():
    return
    
def sampling():
    return
    
if __name__ == "__main__":
   main()
   