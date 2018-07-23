import json
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle

labels = []
embs = []
class_names = []
with open('class.txt') as file:
    for l in file.readlines():
        class_names.append(l.replace('\n', ''))
file.close()

print(class_names)
with open('data.txt') as json_file:  
    data = json.load(json_file)
    for p in data['person']:
        embs.append(p['emb'])
        labels.append(p['name'])
embs = np.array(embs)
labels = np.array(labels)
print(labels)
X_train, X_test, y_train, y_test = train_test_split(embs, labels, test_size=0.33, random_state=42)
print('Training SVM classifier')
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

predictions = model.predict_proba(X_test)
best_class_indices = np.argmax(predictions, axis=1)
best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
print(best_class_probabilities)
accuracy = np.mean(np.equal(best_class_indices, y_test))
print('Accuracy: %.3f' % accuracy)
with open('svm_classifier.pkl', 'wb') as outfile:
    pickle.dump((model, class_names), outfile)



from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors = 4)
neigh.fit(X_train, y_train) 
predictions = neigh.predict_proba(X_test)
best_class_indices = np.argmax(predictions, axis=1)
best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
print(best_class_probabilities)
accuracy = np.mean(np.equal(best_class_indices, y_test))
print('Accuracy: %.3f' % accuracy)
with open('knn_classifier.pkl', 'wb') as outfile:
    pickle.dump((neigh, class_names), outfile)

X_reduced = TruncatedSVD(n_components=50, random_state=0).fit_transform(embs)
X_embedded = TSNE(n_components=2, perplexity=40, verbose=2).fit_transform(X_reduced)

fig = plt.figure(figsize=(10, 10))
ax = plt.axes(frameon=False)
plt.setp(ax, xticks=(), yticks=())
plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9,
                wspace=0.0, hspace=0.0)
plt.scatter(X_embedded[:, 0], X_embedded[:, 1],
        c=labels, marker="x")
plt.show()
pca = PCA(n_components=2)
pca.fit(embs)
embs = pca.transform(embs[1:100])
plt.scatter(embs[:, 0], embs[:, 1],
        c=labels[1:100], marker="x")
plt.show()
