import os
import random
import shutil # libreria con metodi utili alla copia di file di cartella in cartella
import piexif # libreria con metodi per la manipolazione di file in formato exif

def train_test_split(src_folder, train_size = 0.8):
	# Ci assicuriamo che le cartelle esistenti vengano rimosse in modo da poter partire da uno stato pulito
	shutil.rmtree(src_folder+'Train/Cat/', ignore_errors=True)
	shutil.rmtree(src_folder+'Train/Dog/', ignore_errors=True)
	shutil.rmtree(src_folder+'Test/Cat/', ignore_errors=True)
	shutil.rmtree(src_folder+'Test/Dog/', ignore_errors=True)

	# Creazione delle cartelle Train e Test
	os.makedirs(src_folder+'Train/Cat/')
	os.makedirs(src_folder+'Train/Dog/')
	os.makedirs(src_folder+'Test/Cat/')
	os.makedirs(src_folder+'Test/Dog/')

	# Estrazione numero delle immagini di cani e gatti e rimozione file corrotti
	_, _, cat_images    = next(os.walk(src_folder+'Cat/'))
	files_to_be_removed = ['Thumbs.db', '666.jpg', '835.jpg','140.jpg','660.jpg',
						   '850.jpg', '936.jpg', '2663.jpg', '3300.jpg', '3491.jpg', '4833.jpg', '8470.jpg', '5553.jpg',
						   '7964.jpg', '7968.jpg', '7978.jpg', '9171.jpg', '9565.jpg', '9778.jpg', '10125.jpg',
						   '10404.jpg', '10501.jpg', '10820.jpg', '11210.jpg', '11565.jpg', '11874.jpg', '11935.jpg'] # alcuni file danneggiati contenuti nel dataset che davano problemi in fase di test
	for file in files_to_be_removed:
		cat_images.remove(file)
	num_cat_images       = len(cat_images)
	num_cat_images_train = int(train_size * num_cat_images)
	num_cat_images_test  = num_cat_images - num_cat_images_train

	_, _, dog_images    = next(os.walk(src_folder+'Dog/'))
	files_to_be_removed = ['Thumbs.db', '11702.jpg', '1308.jpg', '1866.jpg', '2384.jpg', '10401.jpg', '10797.jpg',
						   '2688.jpg', '2877.jpg', '3136.jpg', '3288.jpg', '3588.jpg', '5604.jpg', '7369.jpg', '4367.jpg',
						   '5736.jpg', '6059.jpg', '6238.jpg', '7112.jpg', '7133.jpg', '7459.jpg', '7969.jpg', '10158.jpg',
						   '8730.jpg', '9188.jpg', '10747.jpg', '11410.jpg', '11675.jpg', '11849.jpg', '11853.jpg', '6718.jpg'] # alcuni file danneggiati contenuti nel dataset che davano problemi in fase di test
	for file in files_to_be_removed:
		dog_images.remove(file)
	num_dog_images       = len(dog_images)
	num_dog_images_train = int(train_size * num_dog_images)
	num_dog_images_test  = num_dog_images - num_dog_images_train

	# Assegnazione in maniera casuale le immagini alle cartelle Train e Test
	cat_train_images = random.sample(cat_images, num_cat_images_train)
	for img in cat_train_images:
		shutil.copy(src=src_folder+'Cat/'+img, dst=src_folder+'Train/Cat/')
	cat_test_images  = [img for img in cat_images if img not in cat_train_images]
	for img in cat_test_images:
		shutil.copy(src=src_folder+'Cat/'+img, dst=src_folder+'Test/Cat/')

	dog_train_images = random.sample(dog_images, num_dog_images_train)
	for img in dog_train_images:
		shutil.copy(src=src_folder+'Dog/'+img, dst=src_folder+'Train/Dog/')
	dog_test_images  = [img for img in dog_images if img not in dog_train_images]
	for img in dog_test_images:
		shutil.copy(src=src_folder+'Dog/'+img, dst=src_folder+'Test/Dog/')

	# Rimozione dati corrotti dal dataset
	remove_exif_data(src_folder+'Train/')
	remove_exif_data(src_folder+'Test/')

# Funzione di supporto per la rimozione dei dati corrotti dal dataset
def remove_exif_data(src_folder):
	_, _, cat_images = next(os.walk(src_folder+'Cat/'))
	for img in cat_images:
		try:
			piexif.remove(src_folder+'Cat/'+img)
		except:
			pass

	_, _, dog_images = next(os.walk(src_folder+'Dog/'))
	for img in dog_images:
		try:
			piexif.remove(src_folder+'Dog/'+img)
		except:
			pass