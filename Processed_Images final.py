#%%
# 전처리부
from PIL import Image
import cv2
import os
import csv
import numpy as np

# 호모모픽 필터를 위한 함수
def homomorphic_filter(img):
    img_YUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y = img_YUV[:,:,0]

    rows, cols = y.shape[:2]
    imgLog = np.log1p(np.array(y, dtype="float") / 255)

    M, N = 2*rows+1, 2*cols+1
    sigma = 10
    (X, Y) = np.meshgrid(np.linspace(0, N-1, N), np.linspace(0, M-1, M))
    Xc, Yc = np.ceil(N/2), np.ceil(M/2)
    gaussianNumerator = (X - Xc)**2 + (Y - Yc)**2

    LPF = np.exp(-gaussianNumerator / (2*sigma*sigma))
    HPF = 1 - LPF
    LPF_shift, HPF_shift = np.fft.ifftshift(LPF), np.fft.ifftshift(HPF)

    img_FFT = np.fft.fft2(imgLog, (M, N))
    img_LF = np.real(np.fft.ifft2(img_FFT * LPF_shift, (M, N)))
    img_HF = np.real(np.fft.ifft2(img_FFT * HPF_shift, (M, N)))

    gamma1, gamma2 = 0.3, 1.5
    img_adjusting = gamma1*img_LF[:rows, :cols] + gamma2*img_HF[:rows, :cols]

    img_exp = np.expm1(img_adjusting)
    img_exp = (img_exp - np.min(img_exp)) / (np.max(img_exp) - np.min(img_exp))
    img_out = np.array(255*img_exp, dtype='uint8')

    img_YUV[:,:,0] = img_out
    result = cv2.cvtColor(img_YUV, cv2.COLOR_YUV2BGR)
    return result

# 고기 추출 함수 정의
def extract_meat(img):
    # HSV 색상 공간으로 변환
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 고기의 붉은색 부분에 대한 HSV 범위 정의
    lower_red1 = np.array([0, 50, 50]) 
    upper_red1 = np.array([10, 255, 255]) 
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # 빨간색 부분을 마스킹
    mask_red1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # 가우시안 블러와 모폴로지 연산을 사용하여 마스크 정제
    blurred_mask_red = cv2.GaussianBlur(mask_red, (9, 9), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    morphed_mask_red = cv2.morphologyEx(blurred_mask_red, cv2.MORPH_CLOSE, kernel)
    morphed_mask_red = cv2.morphologyEx(morphed_mask_red, cv2.MORPH_OPEN, kernel)

    # 마스크로부터 컨투어 찾기
    contours, _ = cv2.findContours(morphed_mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 고기로 추정되는 모든 컨투어를 선택하여 하나의 마스크로 결합
    contour_mask = np.zeros_like(morphed_mask_red)
    for contour in contours:
        if cv2.contourArea(contour) > 500: 
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
    
    # 컨투어 마스크를 원본 이미지에 적용하여 결과 이미지 생성
    result_img = cv2.bitwise_and(img, img, mask=contour_mask)
    
    # 수정된 부분: 처리된 이미지와 마스크 둘 다 반환
    return result_img, contour_mask

def convert_to_grayscale(img):
    # OpenCV를 사용하여 이미지를 그레이스케일로 변환합니다.
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img

def draw_largest_inscribed_rectangle_and_remove_outside(img, mask, shrink_factor=0.6):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_area = 0
    largest_rect = None

    for contour in contours:
        rect = cv2.minAreaRect(contour)  # 회전된 사각형을 얻습니다.
        box = cv2.boxPoints(rect)  # 회전된 사각형의 꼭지점을 얻습니다.
        box = np.intp(box)
        
        area = cv2.contourArea(box)
        if area > largest_area:
            largest_area = area
            largest_rect = box

    if largest_rect is not None:
        # 사각형의 중심을 계산합니다.
        center = np.mean(largest_rect, axis=0)
        
        # 각 꼭지점을 중심으로 이동시킨 후, shrink_factor를 곱하여 크기를 조정합니다.
        new_rect = (largest_rect - center) * shrink_factor + center
        new_rect = np.intp(new_rect)

        # 조정된 사각형을 기반으로 이미지 내부만 남깁니다.
        mask = np.zeros_like(img[:, :, 0])  # 새로운 마스크를 생성합니다.
        cv2.drawContours(mask, [new_rect], 0, (255), -1)  # 사각형 내부를 채웁니다.
        
        # 마스크를 적용하여 이미지의 해당 부분만 추출합니다.
        new_img = np.zeros_like(img)
        for i in range(3):  # BGR 채널 모두에 대해 적용
            new_img[:,:,i] = cv2.bitwise_and(img[:,:,i], img[:,:,i], mask=mask)

    else:
        new_img = img  # 사각형이 없는 경우 원본 이미지 반환

    return new_img

# 경로 설정
preprocessed_dir = '../processtoblackFinal'
labels_csv_path = '../processtoblackFinal/labels.csv'
base_dir = '../../KMeat/KMeat(12.20)'

sub_dirs = ['Duengsim', 'ChaeGGuet']

# 'processed_images_extracted' 폴더 생성
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
            img = cv2.imread(img_path)

            # 호모모픽 필터 적용
            img_filtered = homomorphic_filter(img)
            # 고기 추출 및 마스크 얻기
            meat_img, meat_mask = extract_meat(img_filtered)  
            # 가장 큰 내접 사각형 그리기
            img_with_rectangle = draw_largest_inscribed_rectangle_and_remove_outside(meat_img, meat_mask)
            # 흑백화
            meat_img_contrast = convert_to_grayscale(img_with_rectangle)

            meat_img_pil = Image.fromarray(cv2.cvtColor(meat_img_contrast, cv2.COLOR_BGR2RGB))
            save_name = os.path.splitext(img_name)[0] + '.jpg'
            save_path = os.path.join(save_dir_path, save_name)
            meat_img_pil.save(save_path, 'JPEG')
            
            labels_writer.writerow([save_name, sub_dir])  
# %%
