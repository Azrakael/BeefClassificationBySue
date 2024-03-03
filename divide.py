#%%
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import shutil

# CSV 파일 로드
labels_df = pd.read_csv('../extracted_fat/labels.csv')

# 새로운 폴더 경로 설정
dataset_dir = './dataset'
train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'val')
test_dir = os.path.join(dataset_dir, 'test')

# 필요한 폴더 생성
for folder in [train_dir, val_dir, test_dir]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# 각 클래스별로 데이터를 분할하는 함수
def split_data(class_name, df, source_dir, destination_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # 해당 클래스에 해당하는 이미지만 필터링
    class_df = df[df['Label'] == class_name]

    # train, val, test로 분할 (먼저 train과 나머지로 분할한 후, 나머지를 val과 test로 분할)
    train_df, temp_df = train_test_split(class_df, train_size=train_ratio, random_state=42)
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)  # 전체 대비 val 세트의 비율 조정
    val_df, test_df = train_test_split(temp_df, train_size=val_ratio_adjusted, random_state=42)

    # 이미지 파일을 각 폴더로 이동
    for df, subfolder in zip([train_df, val_df, test_df], ['train', 'val', 'test']):
        subfolder_path = os.path.join(destination_dir, subfolder)
        for img_name in df['Image Name']:
            src_path = os.path.join(source_dir, class_name, img_name)
            dst_path = os.path.join(subfolder_path, img_name)
            shutil.copy(src_path, dst_path)

# 원본 이미지 파일이 있는 폴더 경로 설정
source_dir = '../extracted_fat'

# 각 클래스명에 대해 split_data 함수 호출
for class_name in labels_df['Label'].unique():
    split_data(class_name, labels_df, source_dir, dataset_dir)
# %%
