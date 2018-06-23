import json
import sys
import argparse
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


labels = []
embs = []
file = open('class.txt','r')  
class_names = file.read()
file.close()
print(class_names)
with open('data.txt') as json_file:  
    data = json.load(json_file)
    for p in data['person']:
        embs.append(p['emb'])
        labels.append(p['name'])
embs = np.array(embs)
labels = np.array(labels)
def tSNE():
    X_reduced = TruncatedSVD(n_components=50, random_state=0).fit_transform(embs)
    X_embedded = TSNE(n_components=2, perplexity=40, verbose=2).fit_transform(X_reduced)

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(frameon=False)
    plt.setp(ax, xticks=(), yticks=())
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9,wspace=0.0, hspace=0.0)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1],c=labels, marker="x")
    plt.show()
def PCA():
    pca = PCA(n_components=3)
    pca.fit(embs)
    embs = pca.transform(embs)
    ax = fig.add_subplot(111, projection='3d')
    plt.scatter(embs[:, 0], embs[:, 1],embs[:,2],c=labels, marker="x")
    plt.show()
def main(args):
    if argv.mode == 'tSNE':
        tSNE()
    else:
        PCA()   
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['tSNE', 'PCA'],help='Visualize the embedded vector') 
    return parser.parse_args(argv)
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
