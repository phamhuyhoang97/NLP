import pickle
import numpy as np
from pyvi import ViTokenizer
import nltk
from gensim.models import KeyedVectors

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from sklearn.cluster import AgglomerativeClustering

Number_line = 6

############## Load file .txt
with open('test_data.txt', 'r') as fp:
    contents = fp.read()

print(contents)

## ========= Pre-processing: removing uppercase letter, space of passages 0 to 10
contents_parsed = contents.lower().strip()
print(contents_parsed)
# print(contents_parsed)


## ======== separation of sentences of passage 3
nltk.download('punkt')
sentences = nltk.sent_tokenize(contents_parsed)
# print(sentences)

## ======== use Work2Vec to change sentence to vector
w2v = KeyedVectors.load_word2vec_format("vi_txt/vi.vec")
vocab = w2v.wv.vocab
X = []
for sentence in sentences:
    sentence = ViTokenizer.tokenize(sentence)
    words = sentence.split(" ")
    sentence_vec = np.zeros((100))
    for word in words:
        if word in vocab:
            sentence_vec+=w2v.wv[word]
    X.append(sentence_vec)

## ======== clustering Kmean =========
n_clusters_kmean = Number_line
kmeans = KMeans(n_clusters=n_clusters_kmean)
kmeans = kmeans.fit(X)
print(kmeans.labels_)

# ======== Determining the closest point to the center
avg = []
for j in range(n_clusters_kmean):
    idx = np.where(kmeans.labels_ == j)[0]
    avg.append(np.mean(idx))

closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
ordering = sorted(range(n_clusters_kmean), key=lambda k: avg[k])
summary = ' '.join([sentences[closest[idx]] for idx in ordering])

print("\n*** Using Kmean clustering:\n")
print(summary)

## ======== hierarchical clustering =========
n_clusters_hierarchy = Number_line  # number of regions
ward = AgglomerativeClustering(n_clusters=n_clusters_hierarchy)
ward = ward.fit(X)
print(ward.labels_)

# ======== Determining the closest point to the center of each cluster
X_Clusters = []
idx_cluster = [] 
for j in range(n_clusters_kmean):
    idx = np.where(ward.labels_ == j)[0]  # List of element in each cluster
    cluster_elements = []
    for k in range(len(idx)):
        cluster_elements.append(X[idx[k]])
    idx_cluster.append(idx)
    X_Clusters.append(cluster_elements)

hierarchical_summary = []
avg1 = []
for j in range(len(X_Clusters)):
    kmeans1 = KMeans(n_clusters = 1)
    kmeans1 = kmeans1.fit(X_Clusters[j])

    idx1 = np.where(kmeans1.labels_ == 0)[0]
    avg1.append(np.mean(idx1))

    closest1, _ = pairwise_distances_argmin_min(kmeans1.cluster_centers_, X_Clusters[j])
    hierarchical_summary.append(' ' + (sentences[idx_cluster[j][closest1[0]]]))

ordering1 = sorted(range(n_clusters_hierarchy), key=lambda k: avg1[k])
sumary2 = ' '.join([sentences[closest[idx]] for idx in ordering])

print("\n*** Using Hierarchical clustering and finding the center cluster by Kmean:\n")
print(sumary2)