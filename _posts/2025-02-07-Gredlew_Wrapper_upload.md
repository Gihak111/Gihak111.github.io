---
layout: single
title:  "리액트 네이티브 Gradle 명령어 정리"
categories: "ReactNative"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# Gradle
윈도우 CMD 환경과 리눅스, 윈도우 파워쉘 환경에서 Gradle 명령어가 다르다. 따라서, 자신의 환경에 맞는 명령어를 사용해야 한다.

### Windows CMD에서 Gradle Wrapper 실행하기

Unix 기반 명령어인 `./gradlew clean`을 Windows CMD에서 실행하면 오류가 발생한다.

#### 1. **CMD에서 Gradle Wrapper 실행**
```cmd
gradlew clean
```

```Unix
./gradlew clean
```

#### 2. **`gradlew` 파일 실행 불가 문제 해결**
Gradle Wrapper(`gradlew`) 파일이 실행되지 않는다면 다음 단계를 수행한다.

- **실행 권한 부여:**  
  Windows CMD에서는 실행 권한 부여가 필요하지 않지만, 파일이 정상적으로 실행되지 않는다면 **Git Bash** 또는 **WSL**에서 아래 명령어를 실행하여 권한을 부여한다.
  ```bash
  chmod +x gradlew
  ```

#### 3. **PowerShell에서 실행**
CMD 대신 PowerShell을 사용할 경우 명령어 앞에 `./`를 붙여야 한다.
```powershell
./gradlew clean
```

#### 4. **Java 환경 변수 확인**
Gradle은 Java를 필요로 하므로 Java가 설치되어 있어야 하며, 환경 변수 `JAVA_HOME`이 올바르게 설정되어 있어야 한다.

- **`JAVA_HOME` 설정 방법:**
  1. **Java 설치 경로 확인:** Java가 설치된 디렉터리(예: `C:\Program Files\Java\jdk-<version>`).
  2. **환경 변수 설정:**
     - CMD에서 다음 명령어 실행:
       ```cmd
       setx JAVA_HOME "C:\Program Files\Java\jdk-<version>"
       ```
     - 또는, **제어판 > 시스템 > 고급 시스템 설정 > 환경 변수**에서 `JAVA_HOME`을 추가한다.

#### 5. **Gradle 버전 확인**
Gradle 버전 정보를 확인하려면 다음 명령어를 실행한다.
```cmd
gradlew -v
```

#### 6. **React Native 실행 명령**
Gradle 작업이 완료되면 프로젝트 루트 디렉터리로 돌아가 React Native 앱을 실행한다.
```cmd
cd ..
npx react-native run-android
```

---

### 최종 명령어 정리
Windows CMD에서 React Native 프로젝트의 Gradle 작업을 실행하려면 아래 명령어를 순서대로 입력한다.
```cmd
cd C:\yout\locatioon
gradlew clean
cd ..
npx react-native run-android
```

---

### 추가 팁
- **Gradle 캐시 정리:**
  Gradle 캐시를 정리하려면 다음 명령어를 실행한다.
  ```cmd
  gradlew cleanBuildCache
  ```
