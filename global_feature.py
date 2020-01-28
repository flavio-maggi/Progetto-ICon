from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py

# parametri
images_per_class = 12501
fixed_size       = tuple((500, 500))
train_path       = "Dataset/PetImages/Train"
h5_data          = 'Output/data.h5'
h5_labels        = 'Output/labels.h5'
bins             = 8

# funzione per estrarre feature dall'immagine: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# funzione per estrarre feature dall'immagine: Haralick Texture
def fd_haralick(image):
    # converte l'immagine in bianco e nero
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # calcola  l'haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

# funzione per estrarre feature dall'immagine: Color Histogram
def fd_histogram(image, mask=None):
    # converte l'immagine in HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # calcola il color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalizza l'histogram
    cv2.normalize(hist, hist)
    # ritorna l'histogram
    return hist.flatten()

# prende i training labels
train_labels = os.listdir(train_path)

# ordina i training labels
train_labels.sort()
print(train_labels)

# liste vuote per mantenere i feature vectors e i labels
global_features = []
labels          = []


# itera su i dati di training che si trovano nelle sotto-cartelle
for training_name in train_labels:

    dir = train_path + "/" + training_name
    #dir = os.path.join(train_path, training_name)
    current_label = training_name
    #dir = "Dataset/PetImages/Test/Dog"

    # itera su le immagini in ogni sotto-cartella
    for x in range(0, images_per_class+1):

        file_list = os.listdir(dir)
        file_name = str(x) + ".jpg"

        #prob1 = "7369.jpg" # "4367.jpg"
        #if file_name == prob1:
        #    continue

        if file_name in file_list:
            file_dir = dir + "/" + str(x) + ".jpg"
            print(file_dir)
        else:
            continue

        # legge l'immagine e la ridimensiona
        image = cv2.imread(file_dir)
        image = cv2.resize(image, fixed_size)

        # estrazione delle global features
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick = fd_haralick(image)
        fv_histogram = fd_histogram(image)

        # Concatenazione delle global features
        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

        # aggiornamento della lista dei labels e delle feature
        labels.append(current_label)
        global_features.append(global_feature)

    print("Cartella processata: {}".format(current_label))

# encoding dei labels
targetNames = np.unique(labels)
le          = LabelEncoder()
target      = le.fit_transform(labels)
print("[STATUS] training labels encoded...")

# normalizzazione delle feature nel range (0-1)
scaler            = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)
print("[STATUS] feature vector normalized...")

# salvataggio delle feature e dei labels utilizzando HDF5
h5f_data = h5py.File(h5_data, 'w')
h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

h5f_label = h5py.File(h5_labels, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()

print("[STATUS] Fine del training...")