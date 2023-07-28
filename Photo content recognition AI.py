import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# تابع پیش‌پردازش تصویر
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_width, img_height))
    img = img.reshape(1, img_width, img_height, 3)
    img = img / 255.0
    return img

# تابع تشخیص تصویر
def predict_image(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    class_label = list(database.keys())[class_index]
    return class_label

# مشخصات تصاویر و دسته‌بندی‌ها
img_width, img_height = 224, 224
batch_size = 32
epochs = 10

# دیتابیس تصاویر
database = {
    "car": [".jpg", "path/to/car_image2.jpg", ...],
    "بستنی": ["path/to/ice_cream_image1.jpg", "path/to/ice_cream_image2.jpg", ...],
    # دیگر دسته‌ها را هم می‌توانید اضافه کنید
}

# ایجاد مدل شبکه عصبی
base_model = VGG16(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(database), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# آماده‌ساز داده‌ها
datagen = ImageDataGenerator(rescale=1.0/255)
train_data_generator = datagen.flow_from_directory(
    "path/to/training_data_directory",
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# آموزش مدل
model.fit(train_data_generator, epochs=epochs)

# استفاده از مدل برای تشخیص تصاویر
new_image_path = "path/to/new_image.jpg"
result = predict_image(new_image_path)
print("تشخیص داده شده: ", result)
