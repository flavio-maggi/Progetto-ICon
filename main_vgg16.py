import os
import warnings
warnings.filterwarnings("ignore")
from cnn_utils import train_test_split

src = 'Dataset/PetImages/'

# Controlla se il dataset sia stato scaricato.
if not os.path.isdir(src):
    print(""" Dataset non presente nel computer.""")
    quit()


# Crea le cartelle di train e test se non esistono
if not os.path.isdir(src+'train/'):
    train_test_split(src)

from keras.applications.vgg16 import VGG16

from keras.models import Model
from keras.layers import Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator

# Definizione HYPERPARAMETERS
INPUT_SIZE      = 128               # Numero di pixel per la compressione dell'immagine (in questo caso 32x32),
                                    # impostarlo a 48 se l'esecuzione impiega troppo tempo.
BATCH_SIZE      = 16                # Numero dei training samples da usare in ogni mini batch durante la gradient descent,
                                    # più aumenta più aumenta l'accuratezza, ma aumenta anche il tempo per il training.
STEPS_PER_EPOCH = 200               # Numero di iterazioni per training epoch
EPOCHS          = 3                 # Numero di epoch per effettuare il training sui dati

# Creazione del modello vgg16
vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(INPUT_SIZE, INPUT_SIZE, 3))


# Congela i layer già allenati
for layer in vgg16.layers:
    layer.trainable = False

# Aggiunge un fully connected layer con un nodo alla fine della rete neurale
input_     = vgg16.input
output_    = vgg16(input_)
last_layer = Flatten(name='flatten')(output_)
last_layer = Dense(1, activation='sigmoid')(last_layer)
model      = Model(input=input_, output=last_layer)

# Compila l'ultimo layer aggiunto al modello
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

training_data_generator = ImageDataGenerator(rescale=1./255)
testing_data_generator  = ImageDataGenerator(rescale=1./255)

# Training sulla cartella /Train
training_set = training_data_generator.flow_from_directory(src+'Train/',
                                                target_size=(INPUT_SIZE, INPUT_SIZE),
                                                batch_size=BATCH_SIZE,
                                                class_mode='binary')

# Testing sulla cartella /Test
test_set = testing_data_generator.flow_from_directory(src+'Test/',
                                             target_size=(INPUT_SIZE, INPUT_SIZE),
                                             batch_size=BATCH_SIZE,
                                             class_mode='binary')

print("""
      Attenzione: il modello di training VGG16 potrebbe impiegare più di un'ora se Keras non venisse eseguito sulla GPU.
      Se l'esecuzione dovesse essere troppo lunga è bene ridurre il parametro INPUT_SIZE nel codice per velocizzarlo.
      """)

model.fit_generator(training_set, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, verbose=1)

score = model.evaluate_generator(test_set, steps=100)

for idx, metric in enumerate(model.metrics_names):
    print("{}: {}".format(metric, score[idx]))
