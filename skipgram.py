#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
import math,logging
from pprint import  pprint
from time import time
import matplotlib.pyplot as plt
import pickle

class skipgram(object):
    '''
    skipgram model - refer Mikolov et al (2013)
    '''

    def __init__(self, num_graphs, num_subgraphs, learning_rate, win_size, concat_flag, cpath, batch_size, embedding_size, num_negsample, num_steps, corpus):
        
        self.num_graphs     = num_graphs
        self.num_subgraphs  = num_subgraphs
        self.embedding_size = embedding_size
        self.num_negsample  = num_negsample
        self.learning_rate  = learning_rate
        self.win_size       = win_size
        self.concat_flag    = concat_flag
        self.cpath          = cpath
        self.batch_size     = batch_size
        self.num_steps      = num_steps
        self.corpus         = corpus
        self.graph, self.batch_inputs, self.batch_labels, self.normalized_embeddings, self.loss, self.optimizer = self.trainer_initial()

    def trainer_initial(self):

        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            
            wsize = self.win_size
            batch_inputs = tf.placeholder(tf.int32, shape=([None, 2*wsize+1]))
            batch_labels = tf.placeholder(tf.int64, shape=([None, 1]))

            graph_embeddings = tf.Variable(
                    tf.random_uniform([self.num_graphs, self.embedding_size], -0.5 / self.embedding_size, 0.5/self.embedding_size)
                    ,name="graph_embeddings")

            subgraph_embeddings = tf.Variable(
                    tf.random_uniform([self.num_subgraphs, self.embedding_size], -0.5/self.embedding_size, 0.5/self.embedding_size)
                    ,name="subgraph_embeddings")
            
            embed = []
            batch_doc_embedding = tf.nn.embedding_lookup(graph_embeddings, batch_inputs[:,0])
            if self.concat_flag is 1:
                for j in range(1, 2*wsize+1):
                    embed_w = tf.nn.embedding_lookup(subgraph_embeddings, batch_inputs[:,j])
                    embed.append(embed_w)
                weights = tf.Variable(tf.truncated_normal([self.num_subgraphs, (2*wsize+1)*(self.embedding_size)], stddev=1.0 / math.sqrt((2*wsize+1)*(self.embedding_size)))
                                     ,name="weights")
            else:
                embed_w = tf.zeros([self.batch_size, self.embedding_size])            
                for j in range(1, 2*wsize+1):
                    embed_w += tf.nn.embedding_lookup(subgraph_embeddings, batch_inputs[:,j])
                embed.append(embed_w)
                weights = tf.Variable(tf.truncated_normal([self.num_subgraphs, 2*self.embedding_size], stddev=1.0 / math.sqrt(2*self.embedding_size))
                                     ,name="weights")
                
            embed.append(batch_doc_embedding)
            final_embed = tf.concat(embed, 1)
            biases = tf.Variable(tf.zeros(self.num_subgraphs), name="bias")
            
            #negative sampling part
            loss = tf.reduce_mean(
                tf.nn.nce_loss(weights = weights,
                               biases  = biases,
                               labels  = batch_labels,
                               inputs   = final_embed,
                               num_sampled = self.num_negsample,
                               num_classes = self.num_subgraphs,
                               sampled_values = tf.nn.fixed_unigram_candidate_sampler(
                                   true_classes = batch_labels,
                                   num_true = 1,
                                   num_sampled = self.num_negsample,
                                   unique = True,
                                   range_max = self.num_subgraphs,
                                   distortion = 0.75,
                                   unigrams = self.corpus.subgraph_id_freq_map_as_list)#word_id_freq_map_as_list is the
                               # frequency of each word in vocabulary
                               ))

            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.learning_rate,
                                                       global_step, 100000, 0.96, staircase=True) #linear decay over time

            learning_rate = tf.maximum(learning_rate,0.001) #cannot go below 0.001 to ensure at least a minimal learning

            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

            norm = tf.sqrt(tf.reduce_mean(tf.square(graph_embeddings), 1, keep_dims=True))
            normalized_embeddings = graph_embeddings/norm
            
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
            
        print("Shape final embed",final_embed.get_shape().as_list())
        return graph, batch_inputs, batch_labels, normalized_embeddings, loss, optimizer

    def train(self,corpus,batch_size):

        with tf.Session(graph=self.graph,
                       config=tf.ConfigProto(log_device_placement=True,allow_soft_placement=False)) as sess:

            #init = tf.global_variables_initializer()
            sess.run(self.init)

            loss = 0
            loss_list = []
            
            for i in range(self.num_steps):
                t0 = time()
                step = 0
                while corpus.epoch_flag == False:
                    batch_data, batch_labels = corpus.generate_batch_from_file(batch_size)# get (target,context) word_id tuples

                    feed_dict = {self.batch_inputs:batch_data,self.batch_labels:batch_labels}
                    _,loss_val = sess.run([self.optimizer,self.loss],feed_dict=feed_dict)

                    loss += loss_val

                    if step % 100 == 0:
                        if step > 0:
                            average_loss = loss/step
                            logging.info( 'Epoch: %d : Average loss for step: %d : %f'%(i,step,average_loss))
                    step += 1

                corpus.epoch_flag = False
                epoch_time = time() - t0
                logging.info('#########################   Epoch: %d :  %f, %.2f sec.  #####################' % (i, loss/step,epoch_time))
                loss_list.append(loss/step)
                loss = 0

            #done with training
            final_embeddings = self.normalized_embeddings.eval()
            plt.plot(loss_list)
            
            #### Save model here inside this session ####
            #save_path = self.saver.save(sess, "/home/ipsita/BTP/graph2vec/model_ckpt/model.ckpt")
            save_path = self.saver.save(sess, self.cpath + "/model"+str(self.win_size)+".ckpt")
            print("Model saved in path: %s" % save_path)
            sess.close()
            
        return final_embeddings


# In[1]:


class skipgram_test(object):
    '''
    skipgram model - refer Mikolov et al (2013)
    '''

    def __init__(self, num_graphs, learning_rate, win_size, concat_flag, cpath, batch_size, embedding_size, num_negsample, num_steps, corpus):
        
        self.num_graphs     = num_graphs
        self.num_subgraphs  = 0
        self.embedding_size = embedding_size
        self.num_negsample  = num_negsample
        self.learning_rate  = learning_rate
        self.win_size       = win_size
        self.concat_flag    = concat_flag
        self.cpath          = cpath
        self.batch_size     = batch_size
        self.num_steps      = num_steps
        self.corpus         = corpus
        #self.graph = self.trainer_initial()
        
    #     def trainer_initial(self):

    #         graph = tf.Graph()
    #         with graph.as_default():

    #             graph_embeddings_test = tf.Variable(
    #                     tf.random_uniform([self.num_graphs, self.embedding_size], -0.5 / self.embedding_size, 0.5/self.embedding_size))

    #         return graph
    
    def infer(self, base_dir, subgraph_id_freq_map_as_list, corpus, batch_size):
        
        tf.reset_default_graph()
        
        #restore model 
        sess = tf.Session()
        #saver = tf.train.import_meta_graph('/home/ipsita/BTP/graph2vec/model_ckpt/model.ckpt.meta')
        #saver.restore(sess, save_path='/home/ipsita/BTP/graph2vec/model_ckpt/model.ckpt')
       
        print("path1", self.cpath + '/model'+str(self.win_size)+'.ckpt.meta')
        print("path2", self.cpath + '/model'+str(self.win_size)+'.ckpt')
        saver = tf.train.import_meta_graph(self.cpath + '/model'+str(self.win_size)+'.ckpt.meta')
        ###############saver.restore(sess, save_path=self.cpath + '/model'+str(self.win_size)+'.ckpt')
        
        #print("Variables are -")
        #for v in tf.get_default_graph().as_graph_def().node:
        #      print(v.name)
        
        graph = tf.get_default_graph()
        
        wsize = self.win_size
        batch_inputs_test = tf.placeholder(tf.int32, shape=([None, 2*wsize+1]))
        batch_labels_test = tf.placeholder(tf.int64, shape=([None, 1]))

        print("Graph to infer =", self.num_graphs)
        
        graph_embeddings_test = tf.Variable(
            tf.random_uniform([self.num_graphs, self.embedding_size], -0.5 / self.embedding_size, 0.5/self.embedding_size)
            ,name="graph_embeddings_test")
        
        subgraph_embeddings_test = graph.get_tensor_by_name("subgraph_embeddings:0")
        print("subgraph embedding size",subgraph_embeddings_test.get_shape().as_list())
        self.num_subgraphs = subgraph_embeddings_test.get_shape().as_list()[0]  #### NOTE THIS 
        
        embed = []
        batch_doc_embedding_new = tf.nn.embedding_lookup(graph_embeddings_test, batch_inputs_test[:,0])
        
        if self.concat_flag is 1:
            for j in range(1, 2*wsize+1):
                embed_w = tf.nn.embedding_lookup(subgraph_embeddings_test, batch_inputs_test[:,j])
                embed.append(embed_w)
        else:
            embed_w = tf.zeros([self.batch_size, self.embedding_size])            
            for j in range(1, 2*wsize+1):
                embed_w += tf.nn.embedding_lookup(subgraph_embeddings_test, batch_inputs_test[:,j])
            embed.append(embed_w)

        embed.append(batch_doc_embedding_new)
        final_embed_new = tf.concat(embed, 1)
        
        weights_test = graph.get_tensor_by_name("weights:0")
        biases_test = graph.get_tensor_by_name("bias:0")
        
        with open(base_dir + 'subgraph_id2freq.p', 'rb') as fp:
            subgraph_id_freq_map_as_list_loaded = pickle.load(fp)
    
        
        #negative sampling part
        ## TODO check this function ##
        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights = weights_test,
                           biases  = biases_test,
                           labels  = batch_labels_test,
                           inputs   = final_embed_new,
                           num_sampled = self.num_negsample,
                           num_classes = self.num_subgraphs,
                           sampled_values = tf.nn.fixed_unigram_candidate_sampler(
                               true_classes = batch_labels_test,
                               num_true = 1,
                               num_sampled = self.num_negsample,
                               unique = True,
                               range_max = self.num_subgraphs,
                               distortion = 0.75,
                               unigrams = subgraph_id_freq_map_as_list_loaded)            
                           ))
        
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.learning_rate,
                                                   global_step, 100000, 0.96, staircase=True) #linear decay over time

        learning_rate = tf.maximum(learning_rate, 0.001) #cannot go below 0.001 to ensure at least a minimal learning

        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step, var_list=[graph_embeddings_test])

        norm = tf.sqrt(tf.reduce_mean(tf.square(graph_embeddings_test), 1, keep_dims=True))
        normalized_embeddings = graph_embeddings_test/norm

        self.init = tf.global_variables_initializer() ######## Change here
        sess.run(self.init)
        saver.restore(sess, save_path=self.cpath + '/model'+str(self.win_size)+'.ckpt')
        #sess.run()########################################
        
        #init_new = tf.initialize_variables([graph_embeddings_test])
        #sess.run(init_new)
        
        loss_sum = 0
        loss_list = []

        for i in range(self.num_steps):
            t0 = time()
            step = 0
            while corpus.epoch_flag == False:
                batch_data, batch_labels = corpus.generate_batch_from_file_for_test(base_dir, batch_size)# get (target,context) word_id tuples

                feed_dict = {batch_inputs_test:batch_data, batch_labels_test:batch_labels}
                _,loss_val = sess.run([optimizer, loss],feed_dict=feed_dict)

                loss_sum += loss_val

                if step % 100 == 0:
                    if step > 0:
                        average_loss = loss_sum/step
                        logging.info( 'Epoch: %d : Average loss for step: %d : %f'%(i,step,average_loss))
                step += 1

            corpus.epoch_flag = False
            epoch_time = time() - t0
            logging.info('########################   Epoch: %d :  %f, %.2f sec.  #####################' % (i, loss_sum/step,epoch_time))
            loss_list.append(loss_sum/step)
            loss_sum = 0
        
        
        #done with training
        final_embeddings = normalized_embeddings.eval(session=sess)
        sess.close()
        #final_subgraph_embeddings = self.subgraph_embeddings.eval()
        #plt.plot(loss_list)
        
        return final_embeddings

