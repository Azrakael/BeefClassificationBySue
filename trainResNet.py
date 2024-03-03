#%%
!git clone https://github.com/DEEPI-LAB/python-TensorFlow-Tutorials.git

#%%
import pandas as pd
import tensorflow as tf
import os
import pickle
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# 라벨 데이터 로드
labels_df = pd.read_csv('./extracted_fat/labels.csv')

# 기본 이미지 경로 설정
base_dir = './dataset'

models_dir = '../models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# 훈련, 검증, 테스트 데이터 경로 설정
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# 훈련 데이터를 위한 이미지 증강 설정
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=[0.8, 1.2],
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# 검증 및 테스트 데이터 생성기 (리스케일만 적용)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# 데이터 생성기 설정
train_generator = train_datagen.flow_from_dataframe(
    dataframe=labels_df,
    directory=train_dir,
    x_col='Image Name',
    y_col='Label',
    target_size=(299, 299),  
    batch_size=16,
    class_mode='categorical'
)
val_generator = val_datagen.flow_from_dataframe(
    dataframe=labels_df,
    directory=val_dir,
    x_col='Image Name',
    y_col='Label',
    target_size=(299, 299),
    batch_size=16,
    class_mode='categorical'
)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=labels_df,
    directory=test_dir,
    x_col='Image Name',
    y_col='Label',
    target_size=(299, 299),
    batch_size=16,
    class_mode='categorical',
    shuffle=False
)

# ResNet50 모델 구축 및 수정
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(x)  # l2 정규화
x = Dropout(0.4)(x)
predictions = Dense(2, activation='softmax')(x)  # 클래스 개수에 따라 변경
model = Model(inputs=base_model.input, outputs=predictions)

# 조정된 Adam 옵티마이저
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)

# 모델 컴파일
model.compile(optimizer=adam_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# 콜백 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
model_checkpoint = ModelCheckpoint(
    filepath=os.path.join(models_dir, 'onlyFatResNet50_BM.keras'),
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001, verbose=1)

# 모델 훈련
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    epochs=50,
    validation_data=val_generator,
    validation_steps=val_generator.n // val_generator.batch_size,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

# 학습이 완료된 후 모델 저장
model.save(os.path.join(models_dir, 'onlyFatResNet50.keras'))

with open(os.path.join(models_dir, 'onlyFatResNet50.pkl'), 'wb') as file:
    pickle.dump(history.history, file)

# 검증 데이터로 모델 평가
val_loss, val_accuracy = model.evaluate(val_generator, steps=val_generator.n // val_generator.batch_size)
print(f'Validation loss: {val_loss}, Validation accuracy: {val_accuracy}')

# 테스트 데이터로 모델 평가
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.n // test_generator.batch_size)
print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')
# %%
