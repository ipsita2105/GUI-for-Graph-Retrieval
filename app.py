from flask import Flask, flash, redirect, request, jsonify, render_template
import os
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shutil import copyfile

app = Flask(__name__, static_url_path='/static')
app.secret_key = "some key"

import argparse,os,logging,psutil,time
from joblib import Parallel,delayed
from sklearn.model_selection import train_test_split, KFold
import shutil
import sys, json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

graph_path         = './static/saved_graphs/' 

@app.route('/')

def home():

    if os.path.isdir(graph_path):
        shutil.rmtree(graph_path)
        os.mkdir(graph_path)

    # if os.path.isfile(graph_path+'1_1.png'):
    #     os.remove(graph_path + '1_1.png')
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    selected_dataset = request.form['dataset']
    to_train = request.form['train']
    model_selected = request.form['model']
    epochs = int(request.form['num_epochs'])

    #####################################
    # Check for number of epochs
    ####################################
    if epochs <= 0:                           
        return render_template('index.html', prediction_text="Enter number of epochs > 0")

    #####################################
    # Checking file input
    ######################################

    if 'file' not in request.files:
        flash('No file part')
        return render_template('index.html', prediction_text='No file part')
    
    ffile = request.files['file']
    if ffile.filename == '':
        flash('No selected file')
        return render_template('index.html', prediction_text='No selected file')

    #####################################
    # Display input graph
    ######################################   

    new_graph = graph_path + 'file.gexf'
    ffile.save(new_graph)
    g = nx.read_gexf(new_graph)
    f = plt.figure()
    nx.draw(g, ax=f.add_subplot(111))
    save_graph = os.path.join(graph_path, 'file.png')
    f.savefig(save_graph)

    if model_selected == "graph2vec":
        
        from utils_original import get_files
        from make_graph2vec_corpus_original import wlk_relabel_and_dump_memory_version
        from train_utils_original import train_skipgram

        corpus_dir         = "/home/ipsita/BTP/graph2vec/data/kdd_datasets/"+str(selected_dataset)
        batch_size         = 128
        embedding_size     = 512
        num_negsample      = 10
        learning_rate      = 0.3
        wlk_h              = 3
        label_filed_name   = 'Label'
        wl_extn            = 'g2v'+str(wlk_h) 
        base_dir           = '/home/ipsita/BTP/graph2vec/tmp/for_gui/flask_gui/train/'
        os.mkdir(base_dir)

        output_dir = base_dir + "embeddings"
        os.mkdir(output_dir)

        ########### Add the graph to corpus_dir ######################
        graph_files = get_files(dirname=corpus_dir, extn='.gexf', max_files=0)
        new_graph_number = len(graph_files)
        new_graph_name = corpus_dir + '/' + str(len(graph_files))+'.gexf'
        copyfile(graph_path + 'file.gexf', new_graph_name)

        ############ Read the corpus #######################
        graph_files = get_files(dirname=corpus_dir, extn='.gexf', max_files=0)

        ############## WL Labelling #########################
        wlk_relabel_and_dump_memory_version(graph_files, max_h=wlk_h, node_label_attr_name=label_filed_name)

        X = np.array([g+"."+wl_extn for g in graph_files])
        
        os.mkdir(base_dir+'model_ckpt')
        checkpoint_path = base_dir + 'model_ckpt/1'
        os.mkdir(checkpoint_path)

        embedding_fname = train_skipgram(corpus_dir, wl_extn, learning_rate, embedding_size, num_negsample, epochs, 
                                            batch_size, output_dir)
        
        with open(embedding_fname, 'r') as fh:
            graph_embedding_dict = json.load(fh)

        dict_graphs = np.zeros((len(graph_embedding_dict)-1, embedding_size))

        new_graph_key = "/home/ipsita/BTP/graph2vec/data/kdd_datasets/"+str(selected_dataset)+"/"+str(new_graph_number)+".gexf.g2v3"
        i = 0
        for k in graph_embedding_dict.keys():
            if k != new_graph_key:
                dict_graphs[i,:] = graph_embedding_dict[k]
                i = i + 1
    
        new_graph = np.zeros((1, embedding_size))
        new_graph[0, : ] = graph_embedding_dict[new_graph_key]

        result = cosine_similarity(dict_graphs, new_graph)
        result = np.reshape( result, (np.shape(result)[0]))

        sorted_index = sorted(range(np.shape(result)[0]), key=lambda k:result[k], reverse=True)
        sorted_index = sorted_index[0:5]
        maxvalues = []

        id_list = list(range(1, 6))
        for si in sorted_index:
            maxvalues.append(result[si])
    
        for gnum in sorted_index:
            g_temp = nx.read_gexf('/home/ipsita/BTP/graph2vec/data/kdd_datasets/'+str(selected_dataset)+'/'+str(gnum)+'.gexf')
            f2 = plt.figure()
            nx.draw(g_temp, ax=f2.add_subplot(111))
            save_graph2 = os.path.join(graph_path, str(gnum)+'.png')
            f2.savefig(save_graph2)


        ## delete the saved files
        os.remove("/home/ipsita/BTP/graph2vec/data/kdd_datasets/"+str(selected_dataset)+"/"+str(new_graph_number)+".gexf")
        os.remove("/home/ipsita/BTP/graph2vec/data/kdd_datasets/"+str(selected_dataset)+"/"+str(new_graph_number)+".gexf.g2v3")
        shutil.rmtree(base_dir) 
        
        table_data = zip(id_list, sorted_index, maxvalues)


        return render_template('index.html', 
                                table_data = table_data,
                                input_graph_img= save_graph,
                                dataset=selected_dataset,
                                to_train=epochs)
        
    elif model_selected == "graph2vecContext":

        ########################## imports for infer ####################

        from utils import get_files
        from make_graph2vec_corpus import wlk_relabel_and_dump_memory_version
        from  train_utils import train_skipgram, test_skipgram

        corpus_dir         = "/home/ipsita/BTP/graph2vec/data/kdd_datasets/"+str(selected_dataset)
        batch_size         = 128
        embedding_size     = 512
        num_negsample      = 10
        learning_rate      = 0.3
        wlk_h              = 3
        label_filed_name   = 'Label'
        #class_labels_fname = '/home/ipsita/BTP/graph2vec/data/kdd_datasets/'+str(selected_dataset)+'.Labels'
        win_size           = 2
        concat_flag        = 0  #### Add or concat  
        wl_extn            = 'g2v'+str(wlk_h) 


        if to_train == "yes":
            base_dir           = '/home/ipsita/BTP/graph2vec/tmp/for_gui/flask_gui/train/'
            os.mkdir(base_dir)

            output_dir = base_dir + "embeddings"
            os.mkdir(output_dir)

            ########### Add the graph to corpus_dir ######################
            graph_files = get_files(dirname=corpus_dir, extn='.gexf', max_files=0)
            new_graph_number = len(graph_files)
            new_graph_name = corpus_dir + '/' + str(len(graph_files))+'.gexf'
            copyfile(graph_path + 'file.gexf', new_graph_name)

            ############ Read the corpus #######################
            graph_files = get_files(dirname=corpus_dir, extn='.gexf', max_files=0)

            ############## WL Labelling #########################
            wlk_relabel_and_dump_memory_version(graph_files, max_h=wlk_h, node_label_attr_name=label_filed_name)

            X = np.array([g+"."+wl_extn for g in graph_files])
            
            os.mkdir(base_dir+'model_ckpt')
            checkpoint_path = base_dir + 'model_ckpt/1'
            os.mkdir(checkpoint_path)

            embedding_fname = train_skipgram(base_dir, corpus_dir, wl_extn, learning_rate, win_size, concat_flag,
                                                checkpoint_path, X, embedding_size, num_negsample, epochs, 
                                                batch_size, output_dir)

            with open(embedding_fname, 'r') as fh:
                graph_embedding_dict = json.load(fh)

            dict_graphs = np.zeros((len(graph_embedding_dict)-1, embedding_size))


            new_graph_key = "/home/ipsita/BTP/graph2vec/data/kdd_datasets/"+str(selected_dataset)+"/"+str(new_graph_number)+".gexf.g2v3"
            i = 0
            for k in graph_embedding_dict.keys():
                if k != new_graph_key:
                    dict_graphs[i,:] = graph_embedding_dict[k]
                    i = i + 1
        
            new_graph = np.zeros((1, embedding_size))
            new_graph[0, : ] = graph_embedding_dict[new_graph_key]

            print("DICT_GRAPHS SHAPE", np.shape(dict_graphs))
            print("NEW GRAPH SHAPE", np.shape(new_graph))
            
            result = cosine_similarity(dict_graphs, new_graph)
            result = np.reshape( result, (np.shape(result)[0]))

            sorted_index = sorted(range(np.shape(result)[0]), key=lambda k:result[k], reverse=True)
            sorted_index = sorted_index[0:5]
            maxvalues = []

            id_list = list(range(1, 6))
            for si in sorted_index:
                maxvalues.append(result[si])
        
            for gnum in sorted_index:
                g_temp = nx.read_gexf('/home/ipsita/BTP/graph2vec/data/kdd_datasets/'+str(selected_dataset)+'/'+str(gnum)+'.gexf')
                f2 = plt.figure()
                nx.draw(g_temp, ax=f2.add_subplot(111))
                save_graph2 = os.path.join(graph_path, str(gnum)+'.png')
                f2.savefig(save_graph2)


            ## delete the saved files
            os.remove("/home/ipsita/BTP/graph2vec/data/kdd_datasets/"+str(selected_dataset)+"/"+str(new_graph_number)+".gexf")
            os.remove("/home/ipsita/BTP/graph2vec/data/kdd_datasets/"+str(selected_dataset)+"/"+str(new_graph_number)+".gexf.g2v3")
            shutil.rmtree(base_dir) 
            
            table_data = zip(id_list, sorted_index, maxvalues)


            return render_template('index.html', 
                                    table_data = table_data,
                                    input_graph_img= save_graph,
                                    dataset=selected_dataset,
                                    to_train=epochs)
        
        
        elif to_train == "no":
            base_dir           = '/home/ipsita/BTP/graph2vec/tmp/for_gui/flask_gui/'+str(selected_dataset)+'_folder/'
            output_dir         = base_dir + "embeddings"

            
            g_new_path = [graph_path + 'file.gexf']
            corpus_dir_new = graph_path
        
            wlk_relabel_and_dump_memory_version(g_new_path, max_h=wlk_h, node_label_attr_name=label_filed_name)
        
            cpath = base_dir + 'model_ckpt/1'
            epochs_infer = 400
            embedding_fname_test = test_skipgram (base_dir, corpus_dir_new, wl_extn, learning_rate, win_size, concat_flag, 
                                        cpath, g_new_path, embedding_size, num_negsample, epochs_infer, batch_size, output_dir)

            embedding_fname = base_dir+ '/embeddings/'+str(selected_dataset)+'_dims_512_epochs_400_lr_0.3_embeddings.txt'
            with open(embedding_fname, 'r') as fh:
                graph_embedding_dict = json.load(fh)

            with open(embedding_fname_test, 'r') as fh:
                new_graph_embedding_dict = json.load(fh)

            dict_graphs = np.zeros((len(graph_embedding_dict), embedding_size))

            i = 0
            for k in graph_embedding_dict.keys():
                dict_graphs[i,:] = graph_embedding_dict[k]
                i = i + 1
        
            new_graph = np.zeros((1, embedding_size))

            i = 0
            for k in new_graph_embedding_dict.keys():
                new_graph[i,:] = new_graph_embedding_dict[k]
                i = i + 1
        
            result = cosine_similarity(dict_graphs, new_graph)
            result = np.reshape( result, (np.shape(result)[0]))
        
            sorted_index = sorted(range(np.shape(result)[0]), key=lambda k:result[k], reverse=True)
            sorted_index = sorted_index[0:5]
            maxvalues = []

            id_list = list(range(1, 6))
            for si in sorted_index:
                maxvalues.append(result[si])
        
            for gnum in sorted_index:
                g_temp = nx.read_gexf('/home/ipsita/BTP/graph2vec/data/kdd_datasets/'+str(selected_dataset)+'/'+str(gnum)+'.gexf')
                f2 = plt.figure()
                nx.draw(g_temp, ax=f2.add_subplot(111))
                save_graph2 = os.path.join(graph_path, str(gnum)+'.png')
                f2.savefig(save_graph2)


            ## delete the saved graph
            os.remove(embedding_fname_test)
            #os.remove(graph_path + '1.png')
            os.remove(graph_path + 'file.gexf')
            os.remove(graph_path + 'file.gexf.g2v3')
            
            table_data = zip(id_list, sorted_index, maxvalues)

            return render_template('index.html', 
                                    table_data = table_data,
                                    input_graph_img= save_graph,
                                    dataset=selected_dataset,
                                    to_train=epochs)


@app.route("/display_image")
def display_image():
    return "rose.jpg"

if __name__ == "__main__":
    app.run(debug=True)
