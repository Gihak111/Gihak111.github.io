---
layout: single
title:  "openpose"
categories: "pynb"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# openpose
사람의 뼈대를 딸 수 있는 AI 모델이다.  
설치하는게 생각보다 쉽지 않지만, 할만 하다.
https://github.com/CMU-Perceptual-Computing-Lab/openpose  
위 링크의 깃허브에 올라와 있는 파일을 클론하고, getModels.bat을 실행해도, 서버가 내려가져 있어서 모델이 다운되지 않는 상황이다.  
Cmake를 통해 빌드하는 방법과, 다른 사람이 올려놓은, 드라이브 링크를 통해 모델을 다운받을 수 있다.  
https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/2233
위 링크에서 상황을 볼 수 있다. 아직 서버가 복구되지 않았음을 알 수 있다.  
글 중간에 있는 드라이브를 통해 모델을 받고, 폴더 안에 집어넣는 것으로 사용할 수 있다.
하지만, 여전히 import opennpose를 사용하면 오류가 난다.
이를 해결하기 위해 다음과 같은 형태의 코드로 openpose 모델을 사용할 수 있다.

```python
import subprocess
import os

# OpenPose 실행 경로
openpose_path = r"C:\Users\openpose\bin\OpenPoseDemo.exe"

# 사진 파일 경로 및 출력 디렉토리 설정
image_path = r"C:\Users\images\input_image.jpg"
output_image_dir = r"C:\Users\images\output_images"

# 출력 디렉토리가 존재하지 않으면 생성
if not os.path.exists(output_image_dir):
    os.makedirs(output_image_dir)

# OpenPose 명령어 작성
command = f'cd C:\\Users\\openpose && bin\\OpenPoseDemo.exe --image_dir "{os.path.dirname(image_path)}" --write_images "{output_image_dir}" --display 0 --render_pose 1'

# OpenPose 실행
process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# 명령어 실행 결과 출력
stdout, stderr = process.communicate()
print("STDOUT:", stdout)
print("STDERR:", stderr)

# 실행 결과 확인
if process.returncode == 0:
    print("OpenPose가 성공적으로 실행되었습니다.")
else:
    print("OpenPose 실행 중 오류가 발생했습니다.")


```

위의 코드처럼 파일의 링크를 직접 집어넣는 것으로 모델을 사용할 수 있다.
온라인에 돌아다니는 뉴진스 하니 사진을 활용하여 위의 모델을 돌려보면, 다음과 같은 결과를 얻을 수 있다.  
![입력 사진](https://github.com/user-attachments/assets/c3600d63-dc26-4d8d-9b12-40e65e46ff88)  
![출력 사진](https://github.com/user-attachments/assets/6f490615-194d-4ca2-bcf5-58ed268becf7)  