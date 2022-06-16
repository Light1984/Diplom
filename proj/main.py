import cv2
import numpy as np
import pytesseract
from pathlib import Path
import tensorflow as tf
import os
import random
from tkinter import *
import PIL
from PIL import Image, ImageDraw


#создание папок для букв
print('Creating Folders...')
letters_pack = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
for i in range(len(letters_pack)):
	Path("/Users/romabruks/PycharmProjects/Diplom/train/" + letters_pack[i]).mkdir(parents=True, exist_ok=True)
	Path("/Users/romabruks/PycharmProjects/Diplom/valid/" + letters_pack[i]).mkdir(parents=True, exist_ok=True)
c = random.randint(0, 4)
for l in range(1):
	if l == 1:
		window_range_x = 100
		window_range_y = 100
	elif l == 11 or l == 13:
		window_range_x = 126
		window_range_y = 126
	elif l == 7 or l == 2 or l == 15:
		window_range_x = 130
		window_range_y = 130
	elif l == 8 :
		window_range_x = 140
		window_range_y = 180
	elif l >= 18:
		window_range_x = 110
		window_range_y = 160

	else:
		window_range_x = 120
		window_range_y = 120
	print('Example Number - ' + str(l+1))
	# Загрузка изображения
	print('Uploading...')
	img = cv2.imread('/Users/romabruks/PycharmProjects/Diplom/Прописи/'+str(l+1)+'.jpeg')


	# перевод изборажения в черно-белое
	print('Converting to Black-n-White')
	new_img = np.zeros((img.shape[0],img.shape[1],img.shape[2]), dtype=np.uint8)
	height = img.shape[0]
	width = img.shape[1]
	channels = img.shape[2]


	for i in range(height):
		for j in range(width):
			if img[i,j,0] >= 100 and img[i,j,1] >= 100 and img[i,j,2] >= 100:
				new_img[i, j, 0] = 255
				new_img[i, j, 1] = 255
				new_img[i, j, 2] = 255
			else:
				new_img[i,j,0] = 0
				new_img[i,j,1] = 0
				new_img[i,j,2] = 0
	cv2.imshow('', new_img)
	cv2.waitKey(0)

	# cv2.imshow('',new_img)
	# cv2.waitKey(0)

	# поиск и сохранение букв с помощью окна
	print('Finding Letters')
	window = np.zeros((window_range_x,window_range_y,3))
	letters = np.zeros((0,2))
	ko = 0
	pok = 0
	for i in range(height-window_range_x+1):
		for j in range(width-window_range_y+1):
			window = new_img[i:i+window_range_x-1,j:j+window_range_y-1,:]
			if np.mean(window) < 255.0 * 99 / 100.0 \
					and np.mean(new_img[i:i + window_range_x-1, j, :]) == 255 \
					and np.mean(new_img[i:i + window_range_x-1, j + window_range_y-1, :]) == 255 \
					and np.mean(new_img[i, j:j + window_range_y-1, :]) == 255 \
					and np.mean(new_img[i + window_range_x-1, j:j + window_range_y-1, :]) == 255:
				if letters.shape[0]>0:
					ko = 0
					for k in range(letters.shape[0]):
						if not (letters[k,0]-1 > i or letters[k,0]+window_range_x < i or letters[k,1]-1 > j or letters[k,1]+window_range_y < j)\
							or not (letters[k,0]-1 > i+window_range_x-1 or letters[k,0]+window_range_x < i+window_range_x-1
                                    or letters[k,1]-1 > j or letters[k,1]+window_range_y < j)\
							or not (letters[k,0]-1 > i or letters[k,0]+window_range_x < i or letters[k,1]-1 > j+window_range_y-1
                                    or letters[k,1]+window_range_y < j+window_range_y-1)\
							or not (letters[k,0]-1 > i+window_range_x-1 or letters[k,0]+window_range_x < i+window_range_x-1
                                    or letters[k,1]-1 > j+window_range_y-1 or letters[k,1]+window_range_y < j+window_range_y-1):
							ko += 1
					if 	ko == 0:
						letters = np.append(letters,[[i,j]],axis=0)
						print('Letter Number ' + str(letters.shape[0]))
						pok += 1

				else:
					letters = np.append(letters, [[i, j]], axis=0)
					print('First Letter')


	# упорядочивание букв в алфавитном порядке
	print('Alphabet Ordering')
	letters_new = np.zeros((0,2))
	tmp = np.zeros((0,2))
	for j in range (8):
		for i in range(4):
			tmp = np.append(tmp, [letters[np.argmin(letters[:, 0]), :]], axis=0)
			letters = np.delete(letters, np.argmin(letters[:, 0]), 0)
		for i in range(4):
			letters_new = np.append(letters_new, [tmp[np.argmin(tmp[:, 1]), :]], axis=0)
			tmp = np.delete(tmp, np.argmin(tmp[:, 1]), 0)
		tmp = np.zeros((0, 2))
	letters_new = np.append(letters_new, [letters[0, :]], axis=0)



	# обрезание изображения и сохранение
	print('Cutting')
	tmp_x = 0
	tmp_y = 0
	for i in range(letters_new.shape[0]):
		tmp_x = letters_new[i,0]+window_range_x-1
		tmp_y = letters_new[i,1]+window_range_y-1
		while np.mean(new_img[int(letters_new[i,0]), int(letters_new[i,1]):int(tmp_y), :]) == 255:
			letters_new[i,0] += 1
		while np.mean(new_img[int(letters_new[i,0]):int(tmp_x), int(letters_new[i,1]), :]) == 255:
			letters_new[i,1] += 1
		while np.mean(new_img[int(tmp_x), int(letters_new[i,1]):int(tmp_y), :]) == 255:
			tmp_x -= 1
		while np.mean(new_img[int(letters_new[i,0]):int(tmp_x), int(tmp_y), :]) == 255:
			tmp_y -= 1

		download_image = new_img[int(letters_new[i,0]):int(tmp_x), int(letters_new[i,1]):int(tmp_y), :]
		download_image = cv2.resize(download_image, (100,100), interpolation=cv2.INTER_AREA)

		if (l+1)%5 == c:
			cv2.imwrite('/Users/romabruks/PycharmProjects/Diplom/valid/' + letters_pack[i] + '/'  + str(l+1) +'.jpg',download_image)
		else:
			cv2.imwrite('/Users/romabruks/PycharmProjects/Diplom/train/' + letters_pack[i] + '/'  + str(l + 1) + '.jpg', download_image)



# обучение

root_path = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
TRAINING_DIR = "/Users/romabruks/PycharmProjects/Diplom/train/"
train_datagen = ImageDataGenerator(rescale=1.0 / 255.)
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
							  batch_size=40,
							  class_mode='binary',
							  target_size=(278,278))
#
VALIDATION_DIR = "/Users/romabruks/PycharmProjects/Diplom/valid/"
validation_datagen = ImageDataGenerator(rescale=1.0 / 255.)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
									  batch_size=40,
									  class_mode='binary',
									  target_size=(278,278))
#
model = tf.keras.models.Sequential([
	tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
						   input_shape=(278,278, 3)),
	tf.keras.layers.MaxPooling2D(2, 2),
	tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
	tf.keras.layers.MaxPooling2D(2, 2),
	tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
	tf.keras.layers.MaxPooling2D(2, 2),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(512, activation='relu'),
	tf.keras.layers.Dense(33, activation='softmax')
])
model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit_generator(train_generator,
						   epochs=20,
						   verbose=1,
						   validation_data=validation_generator
						   )

model.save('model.h5')

# epoch = 5 - 98


# testing

letters = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
image = tf.keras.preprocessing.image
model = tf.keras.models.load_model('model.h5')


def save():
	filename = 'test.png'
	image1.save(filename)

	img = image.load_img("test.png")
	imga = np.array(img)
	x_min = 0
	y_min = 0
	x_max = 479
	y_max = 639
	while np.mean(imga[int(x_min), int(y_min):int(y_max), :]) == 255:
		x_min += 1
	while np.mean(imga[int(x_min):int(x_max), int(y_min), :]) == 255:
		y_min += 1
	while np.mean(imga[int(x_max), int(y_min):int(y_max):]) == 255:
		x_max -= 1
	while np.mean(imga[int(x_min):int(x_max), int(y_max), :]) == 255:
		y_max -= 1
	img = Image.fromarray(imga[int(x_min):int(x_max), int(y_min):int(y_max), :].astype('uint8'), 'RGB')

	img = img.resize((278,278))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	images = np.vstack([x])
	classes = model.predict(images, batch_size=1)
	result = int(np.argmax(classes))
	result = letters[result]
	print(result)

def paint(event):
	x1, y1 = (event.x), (event.y)
	x2, y2 = (event.x + 15), (event.y + 15)
	cv.create_oval((x1, y1, x2, y2), fill='black', width=1)
	draw.ellipse((x1, y1, x2, y2), fill='black', width=1)

def clear_list():
	cv.delete('all')
	draw.line((0, 0, 1000, 1000), fill='white', width=1000)

root = Tk()
cv = Canvas(root, width=640, height=480, bg='white')
image1 = PIL.Image.new('RGB', (640, 480), 'white')
draw = ImageDraw.Draw(image1)
cv.bind('<B1-Motion>', paint)
cv.pack(expand=YES, fill=BOTH)
btn_save = Button(text="save", command=save)
btn_clear = Button(text="clear", command=clear_list)
btn_save.pack()
btn_clear.pack()
root.mainloop()
