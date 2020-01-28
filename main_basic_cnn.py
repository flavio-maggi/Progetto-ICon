import os
import warnings
warnings.filterwarnings("ignore")
from cnn_utils import train_test_split

src = 'Dataset/PetImages/'

# Controlla che il dataset sia stato scaricato
if not os.path.isdir(src):
    print("""Dataset non presente nel computer.""")
    quit()

# Crea le cartelle Train e Test se non esistono
if not os.path.isdir(src+'train/'):
    train_test_split(src)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# Definizione HYPERPARAMETERS
FILTER_SIZE     = 3                     # grandezza del filtro per le convoluzioni (in questo caso 3x3)
NUM_FILTERS     = 32                    # il numero di filtri utilizati
INPUT_SIZE      = 32                    # numero di pixel per la compressione dell'immagine (in questo caso 32x32), ci sarà perdita di informazione ma aumento delle prestazioni generali
MAXPOOL_SIZE    = 2                     # grandezza per max pooling (in questo caso 2x2, dimezzerà l'input del layer precendente)
BATCH_SIZE      = 16                    # numero dei training samples da usare in ogni mini batch durante la gradient descent. Più aumenta più aumenta l'accuratezza, ma aumenta anche il tempo per il training
STEPS_PER_EPOCH = 20000//BATCH_SIZE     # numero di iterazioni per training epoch
EPOCHS          = 10                    # numero di epoch per effettuare il training sui dati

# creazione del modello sequenziale
model = Sequential()

# aggiunta del primo convolutional layer
model.add(Conv2D(NUM_FILTERS, (FILTER_SIZE, FILTER_SIZE), input_shape = (INPUT_SIZE, INPUT_SIZE, 3), activation = 'relu')) # 'relu' serve a specificare ReLU come funzione di attivazione
# aggiunta del primo max pooling layer
model.add(MaxPooling2D(pool_size = (MAXPOOL_SIZE, MAXPOOL_SIZE)))

# aggiunta del secondo convolutional layer
model.add(Conv2D(NUM_FILTERS, (FILTER_SIZE, FILTER_SIZE), activation = 'relu'))
# aggiunta del secondo max pooling layer
model.add(MaxPooling2D(pool_size = (MAXPOOL_SIZE, MAXPOOL_SIZE)))

model.add(Flatten()) # funzione che trasforma un vettore multidimensionale in un vettore a singola dimensione
# aggiunta del primo fully connected layer
model.add(Dense(units = 128, activation = 'relu')) # creazione di 128 nodi e funzione di attivazione ReLU

# aggiunta del dropout layer
model.add(Dropout(0.5))
# aggiunta del secondo fully connected layer
model.add(Dense(units = 1, activation = 'sigmoid')) # creazione di 1 nodo e 'sigmoid' specifica la Sigmoid function come funzione di attivazione

# compilazione del modello
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

training_data_generator = ImageDataGenerator(rescale = 1./255)
testing_data_generator  = ImageDataGenerator(rescale = 1./255)

# training sulla cartella /Train
training_set = training_data_generator.flow_from_directory(src+'Train/',
                                                target_size = (INPUT_SIZE, INPUT_SIZE),
                                                batch_size  = BATCH_SIZE,
                                                class_mode  = 'binary')
# testing sulla cartella /Test
test_set = testing_data_generator.flow_from_directory(src+'Test/',
                                             target_size = (INPUT_SIZE, INPUT_SIZE),
                                             batch_size  = BATCH_SIZE,
                                             class_mode  = 'binary')

model.fit_generator(training_set, steps_per_epoch = STEPS_PER_EPOCH, epochs = EPOCHS, verbose=1)

score = model.evaluate_generator(test_set, steps=100)

for idx, metric in enumerate(model.metrics_names):
    print("{}: {}".format(metric, score[idx]))
