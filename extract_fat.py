#%%
import cv2
import numpy as np
import os
import csv
from PIL import Image

def extract_fat_advanced(img):
    # 그레이스케일로 이미지 변환
    grayscale = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    
    # 히스토그램 이퀄라이제이션을 사용하여 대비 향상
    equalized = cv2.equalizeHist(grayscale)
    
    # 가우시안블러를 통한 노이즈 감소
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    
    # 이진 임계값을 적용하여 이진 이미지 가져오기
    _, thresholded = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    
    # 모폴로지 연산을 위한 커널생성.
    kernel = np.ones((5, 5), np.uint8)
    
    # 모폴로지 연산을 사용하여 노이즈 제거
    opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    
    # 리사이징 전 이미지를 PIL Image 객체로 변환
    opened_image = Image.fromarray(opening)
    
    # 이미지 리사이징 (VGG16에 적합한 크기인 224x224로 조정)
    resized_image = opened_image.resize((224, 224), Image.Resampling.LANCZOS)
    
    return resized_image

# 경로 설정
preprocessed_dir = './extracted_fat'
labels_csv_path = './extracted_fat/labels.csv'
base_dir = '../processtoblack'
sub_dirs = ['Duengsim', 'ChaeGGuet']

# 폴더 생성
if not os.path.exists(preprocessed_dir):
    os.makedirs(preprocessed_dir)

# 라벨 파일 생성
with open(labels_csv_path, 'w', newline='', encoding='utf-8-sig') as labels_file:
    labels_writer = csv.writer(labels_file)
    labels_writer.writerow(['Image Name', 'Label'])

    for sub_dir in sub_dirs:
        full_dir_path = os.path.join(base_dir, sub_dir)
        save_dir_path = os.path.join(preprocessed_dir, sub_dir)

        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)

        for img_name in os.listdir(full_dir_path):
            img_path = os.path.join(full_dir_path, img_name)
            img = Image.open(img_path)
            
            # 전처리 이미지를 생성
            preprocessed_img = extract_fat_advanced(img)
            
            # 전처리된 이미지를 저장
            save_name = os.path.splitext(img_name)[0] + '.jpg'
            save_path = os.path.join(save_dir_path, save_name)
            preprocessed_img.save(save_path, 'JPEG')
            
            # 라벨과 함께 CSV 파일에 경로를 기록
            labels_writer.writerow([save_name, sub_dir])

# %%
