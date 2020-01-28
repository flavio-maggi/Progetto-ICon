import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import os
import random

src = 'Dataset/PetImages/'

# Controlla che il dataset sia stato scaricato
if not os.path.isdir(src):
    print("""Dataset non presente nel computer.""")
    quit()

# Prendi la lista dei nomi dei file
_, _, cat_images = next(os.walk('Dataset/PetImages/Cat'))

# Prepara un plot 3x3 (9 imagini in totale)
fig, ax = plt.subplots(3,3, figsize=(20,10))

# Seleziona casualmente e effettua il plot di un'immagine
for idx, img in enumerate(random.sample(cat_images, 9)):
    img_read = plt.imread('Dataset/PetImages/Cat/'+img)
    ax[int(idx/3), idx%3].imshow(img_read)
    ax[int(idx/3), idx%3].axis('off')
    ax[int(idx/3), idx%3].set_title('Cat/'+img)

plt.show()


# Prendi la lista dei nomi dei file
_, _, dog_images = next(os.walk('Dataset/PetImages/Dog'))

# Prepara un plot 3x3 (9 imagini in totale)
fig, ax = plt.subplots(3,3, figsize=(20,10))

# Seleziona casualmente e effettua il plot di un'immagine
for idx, img in enumerate(random.sample(dog_images, 9)):
    img_read = plt.imread('Dataset/PetImages/Dog/'+img)
    ax[int(idx/3), idx%3].imshow(img_read)
    ax[int(idx/3), idx%3].axis('off')
    ax[int(idx/3), idx%3].set_title('Dog/'+img)

plt.show()