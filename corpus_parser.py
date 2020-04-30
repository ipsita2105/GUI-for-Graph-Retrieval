#!/usr/bin/env python
# coding: utf-8

from nltk.tokenize import RegexpTokenizer
import os
import numpy as np
import logging
import operator
import copy
from collections import defaultdict,Counter
from random import shuffle
from pprint import pprint
import pickle

from utils import get_files

global_subgraph_to_id_map = {}

logger = logging.getLogger()
logger.setLevel("INFO")

class Corpus(object):
    
    def __init__(self, win_size, corpus_folder=None, extn='WL2', max_files=0):
        
        assert corpus_folder != None, "please specify the corpus folder"
        self.corpus_folder = corpus_folder
        self.win_size = win_size
        self.subgraph_index = 1  #########
        self.graph_index = 0
        self.epoch_flag = 0
        self.max_files = max_files
        self.graph_ids_for_batch_traversal = []
        self.extn = extn


    def scan_corpus(self):

        subgraphs = []
        for fname in self.graph_fname_list:
            subgraphs.extend(
                [l.split()[0] for l in open(fname).readlines()])  # just take the first word of every sentence
        subgraphs.append('UNK')

        subgraph_to_freq_map = Counter(subgraphs)
        del subgraphs

        subgraph_to_id_map = {sg: i for i, sg in
                              enumerate(subgraph_to_freq_map.keys())}  # output layer of the skipgram network

        self._subgraph_to_freq_map = subgraph_to_freq_map  # to be used for negative sampling
        self._subgraph_to_id_map = subgraph_to_id_map
        self._id_to_subgraph_map = {v:k for k,v in subgraph_to_id_map.items()}
        self._subgraphcount = sum(subgraph_to_freq_map.values()) #total num subgraphs in all graphs

        self.num_graphs = len(self.graph_fname_list) #doc size
        self.num_subgraphs = len(subgraph_to_id_map) #vocab of word size

        self.subgraph_id_freq_map_as_list = [] #id of this list is the word id and value is the freq of word with corresponding word id
        for i in range(len(self._subgraph_to_freq_map)):
            self.subgraph_id_freq_map_as_list.append(self._subgraph_to_freq_map[self._id_to_subgraph_map[i]])

        return self._subgraph_to_id_map


    def scan_and_load_corpus(self, base_dir, graph_list):
        
        #self.graph_fname_list = get_files(self.corpus_folder, extn=self.extn, max_files=self.max_files)
        self.graph_fname_list = graph_list
        self._graph_name_to_id_map = {g: i for i, g in
                                      enumerate(self.graph_fname_list)}  # input layer of the skipgram network
        self._id_to_graph_name_map = {i: g for g, i in self._graph_name_to_id_map.items()}
        subgraph_to_id_map = self.scan_corpus()
        
        ### Save subgraph2id map in file
        with open(base_dir + 'subgraph2id.p', 'wb') as fp:
            pickle.dump(subgraph_to_id_map, fp, protocol=pickle.HIGHEST_PROTOCOL)
        
        logging.info('number of graphs: %d' % self.num_graphs)
        logging.info('subgraph vocabulary size: %d' % self.num_subgraphs)
        logging.info('total number of subgraphs to be trained: %d' % self._subgraphcount)
        
        print('number of graphs: %d' % self.num_graphs)
        print('subgraph vocabulary size: %d' % self.num_subgraphs)
        print('total number of subgraphs to be trained: %d' % self._subgraphcount)

        self.graph_ids_for_batch_traversal = list(range(self.num_graphs))  ###########################
        shuffle(self.graph_ids_for_batch_traversal)
        
        global global_subgraph_to_id_map
        global_subgraph_to_id_map = {}
        global_subgraph_to_id_map = copy.deepcopy(subgraph_to_id_map)
        
        return self.subgraph_id_freq_map_as_list
        
    def scan_and_load_corpus_for_test(self, graph_list):
        self.graph_fname_list = graph_list
        self._graph_name_to_id_map = {g: i for i, g in enumerate(self.graph_fname_list)}
        self._id_to_graph_name_map = {i: g for g, i in self._graph_name_to_id_map.items()}
        
        self.num_graphs = len(self.graph_fname_list)
        self.graph_ids_for_batch_traversal = list(range(self.num_graphs))  ###########################
        shuffle(self.graph_ids_for_batch_traversal)
        
    def generate_batch_from_file_for_test(self, base_dir, batch_size):
        global global_subgraph_to_id_map
        
        with open(base_dir + 'subgraph2id.p', 'rb') as fp:
            global_subgraph_to_id_map = pickle.load(fp)
            
        target_subgraph_ids = []
        context_subgraph_ids = []
        
        graph_name     = self.graph_fname_list[self.graph_ids_for_batch_traversal[self.graph_index]]
        graph_contents = open(graph_name).readlines()
        while self.subgraph_index >= len(graph_contents)-self.win_size:
            self.subgraph_index = self.win_size
            self.graph_index += 1

            if self.graph_index == len(self.graph_fname_list):   #last graph so wrap around
                self.graph_index = 0
                np.random.shuffle(self.graph_ids_for_batch_traversal)
                self.epoch_flag = True

            graph_name = self.graph_fname_list[self.graph_ids_for_batch_traversal[self.graph_index]]
            graph_contents = open(graph_name).readlines()
            
        while len(context_subgraph_ids) < batch_size:

            while self.subgraph_index >= len(graph_contents)-self.win_size:
                self.subgraph_index = self.win_size
                self.graph_index += 1

                if self.graph_index == len(self.graph_fname_list):
                    self.graph_index = 0
                    np.random.shuffle(self.graph_ids_for_batch_traversal)
                    self.epoch_flag = True

                graph_name = self.graph_fname_list[self.graph_ids_for_batch_traversal[self.graph_index]]
                graph_contents = open(graph_name).readlines()
            
            line_id = self.subgraph_index
            #context_subgraph_1 = graph_contents[line_id].split()[0]
            #context_subgraph_0 = graph_contents[line_id-1].split()[0]
            #context_subgraph_2 = graph_contents[line_id+1].split()[0]
            
            target_graph = graph_name
            context_subgraph_list = []
            context_subgraph_list.append(self._graph_name_to_id_map[target_graph])
            for i in range(-self.win_size, self.win_size + 1):
                if i != 0:
                    #print("win size = ", self.win_size, "i=", i)
                    k = graph_contents[line_id + i].split()[0]
                    if k in global_subgraph_to_id_map:
                        context_subgraph_list.append(global_subgraph_to_id_map[graph_contents[line_id + i].split()[0]] )
                    else:
                        context_subgraph_list.append(global_subgraph_to_id_map['UNK'])
             
            #print("mini list", context_subgraph_list)
            context_subgraph_ids.append(context_subgraph_list)
            
            #context_subgraph_ids.append([self._graph_name_to_id_map[target_graph],
            #                             self._subgraph_to_id_map[context_subgraph_0],
            #                             self._subgraph_to_id_map[context_subgraph_2]
            #                            ])
            #target_subgraph_ids.append(self._subgraph_to_id_map[context_subgraph_1])
            if graph_contents[line_id].split()[0] in global_subgraph_to_id_map:
                target_subgraph_ids.append(global_subgraph_to_id_map[graph_contents[line_id].split()[0]])
            else:
                target_subgraph_ids.append(global_subgraph_to_id_map['UNK'])
            
            self.subgraph_index += 1
        
        #print("context_subgraph_ids shape =",np.shape(context_subgraph_ids) ,"\n", context_subgraph_ids)
        target_subgraph_ids2 = []
        context_subgraph_ids2 = []
        index_shuf = list(range(len(target_subgraph_ids)))

        shuffle(index_shuf)
        for i in index_shuf:
            target_subgraph_ids2.append(target_subgraph_ids[i])
            context_subgraph_ids2.append(context_subgraph_ids[i])

        #target_context_pairs = zip(target_graph_ids, context_subgraph_ids)
        #shuffle(target_context_pairs)
        #target_graph_ids, context_subgraph_ids = zip(*target_context_pairs)
        
        #context_subgraph_ids = []
        target_subgraph_ids = np.array(target_subgraph_ids2, dtype=np.int32)
        context_subgraph_ids = np.array(context_subgraph_ids2, dtype=np.int32)

        targetword_outputs = np.reshape(target_subgraph_ids, [len(target_subgraph_ids), 1])
        #contextword_outputs = np.reshape(context_subgraph_ids, [len(context_subgraph_ids), 1])

        #np.savetxt("batch.txt", (context_subgraph_ids, targetword_outputs), fmt="%d")
        #print("shape of batch context:", np.shape(context_subgraph_ids), "target", np.shape(targetword_outputs))
        return context_subgraph_ids, targetword_outputs
    
    def generate_batch_from_file(self, batch_size):
        target_subgraph_ids = []
        context_subgraph_ids = []
        
        graph_name     = self.graph_fname_list[self.graph_ids_for_batch_traversal[self.graph_index]]
        graph_contents = open(graph_name).readlines()
        while self.subgraph_index >= len(graph_contents)-self.win_size:
            self.subgraph_index = self.win_size
            self.graph_index += 1

            if self.graph_index == len(self.graph_fname_list):   #last graph so wrap around
                self.graph_index = 0
                np.random.shuffle(self.graph_ids_for_batch_traversal)
                self.epoch_flag = True

            graph_name = self.graph_fname_list[self.graph_ids_for_batch_traversal[self.graph_index]]
            graph_contents = open(graph_name).readlines()
            
        while len(context_subgraph_ids) < batch_size:
            

            while self.subgraph_index >= len(graph_contents)-self.win_size:
                self.subgraph_index = self.win_size
                self.graph_index += 1

                if self.graph_index == len(self.graph_fname_list):
                    self.graph_index = 0
                    np.random.shuffle(self.graph_ids_for_batch_traversal)
                    self.epoch_flag = True

                graph_name = self.graph_fname_list[self.graph_ids_for_batch_traversal[self.graph_index]]
                graph_contents = open(graph_name).readlines()
            
            line_id = self.subgraph_index
            #context_subgraph_1 = graph_contents[line_id].split()[0]
            #context_subgraph_0 = graph_contents[line_id-1].split()[0]
            #context_subgraph_2 = graph_contents[line_id+1].split()[0]
            
            target_graph = graph_name
            context_subgraph_list = []
            context_subgraph_list.append(self._graph_name_to_id_map[target_graph])
            for i in range(-self.win_size, self.win_size + 1):
                if i != 0:
                    #print("win size = ", self.win_size, "i=", i)
                    context_subgraph_list.append( self._subgraph_to_id_map[graph_contents[line_id + i].split()[0]] )
             
            #print("mini list", context_subgraph_list)
            context_subgraph_ids.append(context_subgraph_list)
            
            #context_subgraph_ids.append([self._graph_name_to_id_map[target_graph],
            #                             self._subgraph_to_id_map[context_subgraph_0],
            #                             self._subgraph_to_id_map[context_subgraph_2]
            #                            ])
            #target_subgraph_ids.append(self._subgraph_to_id_map[context_subgraph_1])
            target_subgraph_ids.append(self._subgraph_to_id_map[graph_contents[line_id].split()[0]])
            
            self.subgraph_index += 1
        
        #print("context_subgraph_ids shape =",np.shape(context_subgraph_ids) ,"\n", context_subgraph_ids)
        target_subgraph_ids2 = []
        context_subgraph_ids2 = []
        index_shuf = list(range(len(target_subgraph_ids)))

        shuffle(index_shuf)
        for i in index_shuf:
            target_subgraph_ids2.append(target_subgraph_ids[i])
            context_subgraph_ids2.append(context_subgraph_ids[i])

        #target_context_pairs = zip(target_graph_ids, context_subgraph_ids)
        #shuffle(target_context_pairs)
        #target_graph_ids, context_subgraph_ids = zip(*target_context_pairs)
        
        #context_subgraph_ids = []
        target_subgraph_ids = np.array(target_subgraph_ids2, dtype=np.int32)
        context_subgraph_ids = np.array(context_subgraph_ids2, dtype=np.int32)

        targetword_outputs = np.reshape(target_subgraph_ids, [len(target_subgraph_ids), 1])
        #contextword_outputs = np.reshape(context_subgraph_ids, [len(context_subgraph_ids), 1])

        #np.savetxt("batch.txt", (context_subgraph_ids, targetword_outputs), fmt="%d")
        #print("shape of batch context:", np.shape(context_subgraph_ids), "target", np.shape(targetword_outputs))
        return context_subgraph_ids, targetword_outputs


