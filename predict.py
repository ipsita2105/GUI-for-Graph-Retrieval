
#!/usr/bin/env python
# coding: utf-8

import argparse,os,logging,psutil,time
from joblib import Parallel,delayed
from sklearn.model_selection import train_test_split, KFold
import shutil
import sys

from utils import get_files
from make_graph2vec_corpus import *
from  train_utils import train_skipgram
from classify import perform_classification_final_new

import ipywidgets as widgets
from IPython.display import display
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

#######################################################################
base_dir = '/home/ipsita/BTP/graph2vec/tmp/for_gui/flask_gui/'

corpus_dir = "/home/ipsita/BTP/graph2vec/data/kdd_datasets/mutag"
output_dir = base_dir + "embeddings"
batch_size = 128
epochs = 400 
embedding_size = 1024
num_negsample = 10
learning_rate = 0.3
wlk_h = 3
label_filed_name = 'Label'
class_labels_fname = '/home/ipsita/BTP/graph2vec/data/kdd_datasets/mutag.Labels'
win_size = 2
concat_flag = 0  #NOTE: Add or concat

wl_extn = 'g2v'+str(wlk_h) 

assert os.path.exists(corpus_dir), "File {} does not exist".format(corpus_dir)
assert os.path.exists(output_dir), "Dir {} does not exist".format(output_dir)

####################################################################################3


fu = widgets.FileUpload(
    accept='.gexf',  
    multiple=False  
)

##################### INTERATIVE PLOT ############################ 
#plt.ion()

btn = widgets.Button(description='Display')

def btn_eventhandler(obj):
    
    fig = plt.figure(num=1)
    plt.clf()
    x = list(fu.value.keys())[0]
    bin_data = fu.value[x]['content']
    datastr= str(bin_data,'utf-8')
    with open(base_dir+"saved_graphs/1.gexf", "w") as fp:
        fp.write(datastr)
        
    g_result = nx.read_gexf(base_dir+"saved_graphs/1.gexf")
    nx.draw(g_result, pos = nx.nx_pydot.graphviz_layout(g_result))
    #nx.draw(g_result)
    fig.canvas.draw()
    

################ GUI PART ###########################

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')

def home():
    return render_template('./www/layout.html')

@app.route('/predict', methods=['POST'])

def predict():


    g_new_path = [base_dir + '/saved_graphs/1.gexf']
    corpus_dir_new = base_dir + '/saved_graphs/'
    
    wlk_relabel_and_dump_memory_version(g_new_path, max_h=wlk_h, node_label_attr_name=label_filed_name)
    
    cpath = base_dir + 'model_ckpt/1'
    epochs_infer = 400
    embedding_fname_test = test_skipgram (corpus_dir_new, wl_extn, learning_rate, win_size, concat_flag, 
                                     cpath, g_new_path, embedding_size, num_negsample, epochs_infer, batch_size, output_dir)

    embedding_fname = base_dir + '/embeddings/mutag_dims_1024_epochs_400_lr_0.3_embeddings.txt'
    with open(embedding_fname, 'r') as fh:
        graph_embedding_dict = json.load(fh)

    with open(embedding_fname_test, 'r') as fh:
        new_graph_embedding_dict = json.load(fh)

    #print(len(graph_embedding_dict),',' ,len(new_graph_embedding_dict))
    
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
    
    maxvalue = -10
    maxindex = 0

    for i in range(0, len(result)):
        if result[i] > maxvalue:
            maxvalue = result[i]
            maxindex = i 
    
    print("MAXINDEX = ", maxindex+1)

    #g_result = nx.read_gexf('/home/ipsita/BTP/graph2vec/data/kdd_datasets/mutag/'+ str(maxindex+1)+'.gexf')
    
    #Plotting part
    #fig = plt.figure(num=2)
    #plt.clf()
    #nx.draw(g_result, pos = nx.nx_pydot.graphviz_layout(g_result))
    #fig.canvas.draw()
    
    ## delete the saved graph
    os.remove(embedding_fname_test)
    os.remove(base_dir+"saved_graphs/1.gexf")
    os.remove(base_dir+"saved_graphs/1.gexf.g2v3")

    return render_template('layout.html', prediction_text='Nearest molecule is
            {}'.format(output))


    
if __name__ == "__main__":
    app.run(debug=True)















