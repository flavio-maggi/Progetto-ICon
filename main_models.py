import h5py
import numpy as np
import os
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

# parametri
num_trees = 100  # numero degli alberi da creare nel Random Forest Classifier
test_size = 0.10 # parametro da utilizzare nella suddivisione dei dati
seed      = 9    # parametro per i random_state dei classificatori
train_path = "Dataset/PetImages/Train"
h5_data    = 'Output/data.h5'
h5_labels  = 'Output/labels.h5'
scoring    = "accuracy"

# prendi i training labels
train_labels = os.listdir(train_path)

# ordina i training labels
train_labels.sort()

# creazione di tutti i modelli per il machine learning
models = []
models.append(('LR', LogisticRegression(random_state=seed)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state=seed)))
models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=seed)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(random_state=seed)))

# variabili per mantendere i risultati e i nomi
results = []
names   = []

# importa il feature vector e i labels
h5f_data  = h5py.File(h5_data, 'r')
h5f_label = h5py.File(h5_labels, 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string   = h5f_label['dataset_1']

# conversione del file .h5 in un array numpy
global_features = np.array(global_features_string)
global_labels   = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()


# divide i dati per il training e per il testing
(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                          np.array(global_labels),
                                                                                          test_size=test_size,
                                                                                          random_state=seed)

print("\n[STATUS] Inizio della validazione...\n")

# 10-fold cross validation
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

print("\n[STATUS] Stampa dei risultati conclusa!")