---
layout: single
title:  "파이썬으로 방 사진 3D화 하기"
categories: "pynb"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# openCV
다양한 기능을 제공하는 openCV를 활용해서 방 사진을 3D 모델로 바꿔줍니다  
다만 아직 더 수정이 필요하기에 잘못된 결과의 3D 모델이 출력됩니다.  
# 코드
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 이미지 경로 설정
img1_path = r'C:\\1234.jpg'
img2_path = r'C:\\5678.jpg'

# 이미지 경로 확인
if not os.path.exists(img1_path):
    print(f"Error: Image file not found at {img1_path}")
    exit(1)

if not os.path.exists(img2_path):
    print(f"Error: Image file not found at {img2_path}")
    exit(1)
# 이미지 읽기
img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

# 이미지 읽기 오류 확인
if img1 is None:
    print(f"Error: Failed to read image at {img1_path}")
    exit(1)

if img2 is None:
    print(f"Error: Failed to read image at {img2_path}")
    exit(1)
# 이미지 크기를 동일하게 조정
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# SIFT 특징점 추출기 생성
sift = cv2.SIFT_create()

# 특징점과 디스크립터 추출
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# BFMatcher 생성
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# 특징점 매칭
matches = bf.match(descriptors1, descriptors2)

# 매칭 결과를 거리 기준으로 정렬
matches = sorted(matches, key=lambda x: x.distance)

# 매칭 결과를 이미지로 시각화
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img_matches)
plt.show()

# 매칭된 특징점 좌표 추출
pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# 기본 행렬 추정
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

# 내점과 외점을 분리
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

# 깊이 맵 추정을 위한 스테레오 BM 생성
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(img1, img2)

# 깊이 맵 시각화
plt.imshow(disparity, 'gray')
plt.show()

# 3D 포인트 클라우드 생성
h, w = img1.shape
f = 0.8 * w  # 임의의 초점 거리
Q = np.float32([[1, 0, 0, -w / 2],
                [0, -1, 0, h / 2],
                [0, 0, 0, -f],
                [0, 0, 1, 0]])

points_3D = cv2.reprojectImageTo3D(disparity, Q)

# 유효한 깊이 값만 선택
mask = disparity > disparity.min()
out_points = points_3D[mask]
out_colors = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2RGB)[mask]

# 포인트 클라우드 파일 저장
out_fn = 'point_cloud.ply'
with open(out_fn, 'w') as f:
    f.write('ply\n')
    f.write('format ascii 1.0\n')
    f.write('element vertex {}\n'.format(len(out_points)))
    f.write('property float x\n')
    f.write('property float y\n')
    f.write('property float z\n')
    f.write('property uchar red\n')
    f.write('property uchar green\n')
    f.write('property uchar blue\n')
    f.write('end_header\n')
    for p, c in zip(out_points, out_colors):
        f.write('{} {} {} {} {} {}\n'.format(p[0], p[1], p[2], c[0], c[1], c[2]))

print("3D 포인트 클라우드 파일이 저장되었습니다: {}".format(out_fn))

```