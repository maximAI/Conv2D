# Conv2D
Задача: Добится точности не менее 85% (в идеале 90%) на проверочной выборке на базе трех иномарок. Размер проверочной выборки - 20%.

<a name="3"></a>
## [Оглавление:](#3)
1. [Загрузка данных](#1)
2. [Создание сети](#2)


Импортируем нужные библиотеки.
```
from tensorflow.keras.models import Sequential                  # Загружаем абстрактный класс базовой модели сети от кераса
# Подключим необходимые слои
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Подключим ImageDataGenerator для аугментации
from tensorflow.keras.optimizers import Adam, Adadelta          # Подключим оптимизаторы
from tensorflow.keras import utils                              # Подключим utils
from tensorflow.keras.preprocessing import image                # Подключим image для работы с изображениями
from google.colab import files                                  # Подключим гугл диск
import numpy as np                                              # Подключим numpy - библиотеку для работы с массивами данных
import pandas as pd                                             # Загружаем библиотеку Pandas
import matplotlib.pyplot as plt                                 # Подключим библиотеку для визуализации данных
from PIL import Image                                           # Подключим Image для работы с изображениями
import random                                                   # Импортируем библиотеку random
import math                                                     # Импортируем модуль math
import os                                                       # Импортируем модуль os для загрузки данных
%matplotlib inline
```
[:arrow_up:Оглавление](#3)
<a name="1"></a>
## Загрузка данных.
Загрузим базу.
```
!unzip -q "/content/drive/MyDrive/Сверточные сети/middle_fmr.zip" -d /content/drive/MyDrive/cars    # Указываем путь к базе в Google Drive
```
Зададим гиперпараметры.
```
train_path = '/content/drive/MyDrive/cars'  # Папка с папками картинок, рассортированных по категориям
batch_size = 10                     # Размер выборки
img_width = 108                     # Ширина изображения
img_height = 54                     # Высота изображения
```
```
# Генератор изображений
datagen = ImageDataGenerator(
    rescale = 1. / 255,             # Значения цвета меняем на дробные показания
    rotation_range = 10,            # Поворачиваем изображения при генерации выборки
    width_shift_range = 0.1,        # Двигаем изображения по ширине при генерации выборки
    height_shift_range = 0.1,       # Двигаем изображения по высоте при генерации выборки
    shear_range = 0.1,              # Угол сдвига против часовой стрелки в градусах
    channel_shift_range = 0.1,      # Диапазон для случайных сдвигов канала
    zoom_range = 0.1,               # Зумируем изображения при генерации выборки
    cval = 0.1,                     # Значение, используемое для точек за пределами границ
    horizontal_flip = True,         # Отзеркаливание изображений
    fill_mode = 'nearest',          # Заполнение пикселей вне границ ввода
    validation_split = 0.2)         # Указываем разделение изображений на обучающую и тестовую выборку
```
Формирование выборок.
```
train_generator = datagen.flow_from_directory(
    train_path,                             # Путь ко всей выборке выборке
    target_size = (img_width, img_height),  # Размер изображений
    batch_size = batch_size,                # Размер batch_size
    class_mode = 'categorical',             # Категориальный тип выборки. Разбиение выборки по маркам авто
    shuffle = True,                         # Перемешивание выборки
    subset = 'training')                    # Устанавливаем как набор для обучения

validation_generator = datagen.flow_from_directory(
    train_path,                             # Путь ко всей выборке выборке
    target_size = (img_width, img_height),  # Размер изображений
    batch_size = batch_size,                # Размер batch_size
    class_mode = 'categorical',             # Категориальный тип выборки. Разбиение выборки по маркам авто
    shuffle = True,                         # Перемешивание выборки
    subset = 'validation')                  # Устанавливаем как валидационный набор
```

[:arrow_up:Оглавление](#3)
<a name="2"></a>
## Сеть
```
model = Sequential()

model.add(BatchNormalization(input_shape=(img_width, img_height, 3)))
model.add(Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))
model.add(Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (3, 3)))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding = 'same', activation = 'relu'))
model.add(Conv2D(256, (3, 3), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Conv2D(512, (3, 3), padding = 'same', activation = 'relu'))
model.add(Conv2D(1024, (3, 3), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (3, 3)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(2048, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(4096, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(8192, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Dense(len(train_generator.class_indices), activation = 'softmax'))
```
Коспилируем модель и запустим обучение.
```
model.compile(loss = 'categorical_crossentropy', optimizer = Adam(0.0001), metrics = ['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // batch_size,
    epochs = 200,
    verbose = 1)

plt.plot(history.history['accuracy'], 
         label = 'Доля верных ответов на обучающем наборе')
plt.plot(history.history['val_accuracy'], 
         label = 'Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()
```
Дообучим модель.
```
history = model.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // batch_size,
    epochs = 100,
    verbose = 1)

plt.plot(history.history['accuracy'], 
         label = 'Доля верных ответов на обучающем наборе')
plt.plot(history.history['val_accuracy'], 
         label = 'Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()
```
![Иллюстрация к проекту](https://github.com/maximAI/Conv2D/blob/main/Screenshot_1.jpg)

[:arrow_up:Оглавление](#3)

[Ноутбук](https://colab.research.google.com/drive/1Vpx_8v8hZ_Om3MMucuCe8aIE5b-hVfwZ?usp=sharing)
