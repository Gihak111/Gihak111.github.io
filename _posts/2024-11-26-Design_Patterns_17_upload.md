---
layout: single
title:  "디자인 패턴 시리즈 17. 어댑터"
categories: "Design_Patterns"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# 디자인 패턴 시리즈 14: 어댑터 패턴 (Adapter Pattern)

어댑터 패턴(Adapter Pattern)은 호환되지 않는 인터페이스를 가진 클래스들이 함께 동작할 수 있도록 변환해주는 구조 패턴이다.  
쉽게 말해, 서로 다른 인터페이스 사이를 연결해주는 중간 다리 역할을 한다.  

## 어댑터 패턴의 필요성

두 개의 서로 다른 시스템이 함께 동작해야 할 때, 직접적으로 코드를 수정하면 기존 코드에 영향을 미쳐 시스템 안정성이 저하될 수 있다.  
이럴 때 어댑터 패턴을 사용하면 다음과 같은 장점을 얻을 수 있다:

1. 기존 코드 수정 불필요: 기존 클래스나 인터페이스를 수정하지 않고도 원하는 기능을 추가할 수 있다.  
2. 유연성 향상: 기존 코드와 새 코드 간의 연결을 유연하게 처리할 수 있다.  
3. 재사용성 강화: 기존 시스템을 새로운 환경에서 쉽게 재사용할 수 있다.  

## 어댑터 패턴의 구조

1. Target (목표 인터페이스): 클라이언트가 기대하는 인터페이스.  
2. Adaptee (적응 대상): 변환되어야 하는 기존 클래스.  
3. Adapter (어댑터): Target 인터페이스를 구현하며, Adaptee를 감싸서 변환 작업을 수행한다.  

### 구조 다이어그램

```
Client --> Target <---- Adapter ----> Adaptee
```  

## 어댑터 패턴 예시  

### 상황: 미디어 플레이어
MP3 플레이어가 MP4 파일과 같은 다른 포맷도 재생할 수 있도록 만들고 싶다.  
MP3 플레이어는 기존에 제공된 인터페이스를 따르고, MP4 재생 기능은 어댑터를 통해 추가한다.  

### Java로 어댑터 패턴 구현하기

```java
// Target 인터페이스
interface MediaPlayer {
    void play(String audioType, String fileName);
}

// Adaptee 클래스
class AdvancedMediaPlayer {
    void playMP4(String fileName) {
        System.out.println("Playing MP4 file: " + fileName);
    }

    void playVLC(String fileName) {
        System.out.println("Playing VLC file: " + fileName);
    }
}

// Adapter 클래스
class MediaAdapter implements MediaPlayer {
    private AdvancedMediaPlayer advancedMediaPlayer;

    public MediaAdapter(String audioType) {
        if (audioType.equalsIgnoreCase("mp4")) {
            advancedMediaPlayer = new AdvancedMediaPlayer();
        }
    }

    @Override
    public void play(String audioType, String fileName) {
        if (audioType.equalsIgnoreCase("mp4")) {
            advancedMediaPlayer.playMP4(fileName);
        } else if (audioType.equalsIgnoreCase("vlc")) {
            advancedMediaPlayer.playVLC(fileName);
        }
    }
}

// Concrete Target 클래스
class AudioPlayer implements MediaPlayer {
    private MediaAdapter mediaAdapter;

    @Override
    public void play(String audioType, String fileName) {
        if (audioType.equalsIgnoreCase("mp3")) {
            System.out.println("Playing MP3 file: " + fileName);
        } else if (audioType.equalsIgnoreCase("mp4") || audioType.equalsIgnoreCase("vlc")) {
            mediaAdapter = new MediaAdapter(audioType);
            mediaAdapter.play(audioType, fileName);
        } else {
            System.out.println("Invalid media format: " + audioType);
        }
    }
}

// 클라이언트 코드
public class Main {
    public static void main(String[] args) {
        MediaPlayer player = new AudioPlayer();

        player.play("mp3", "song.mp3");
        player.play("mp4", "video.mp4");
        player.play("vlc", "movie.vlc");
        player.play("avi", "unsupported.avi");
    }
}
```  

### 출력 결과  

```
Playing MP3 file: song.mp3
Playing MP4 file: video.mp4
Playing VLC file: movie.vlc
Invalid media format: avi
```  

## 어댑터 패턴의 장점  

1. 호환성 제공: 호환되지 않는 인터페이스 간에 연결을 제공.  
2. 확장성: 기존 코드 수정 없이 새로운 기능을 추가 가능.  
3. 재사용성: 기존 클래스를 새 환경에 맞게 재사용할 수 있음.  

## 어댑터 패턴의 단점  

1. 복잡성 증가: 어댑터를 추가하면 설계가 다소 복잡해질 수 있음.  
2. 성능 문제: 어댑터의 중간 처리로 인해 성능에 약간의 영향을 줄 수 있음.  

## 어댑터 패턴의 실제 사례  

- Java I/O 스트림: `InputStreamReader`는 InputStream을 Reader로 변환하는 어댑터 역할을 한다.  
- GUI 프레임워크: 이벤트 리스너에서 어댑터 클래스를 사용하여 특정 메서드만 오버라이딩하는 경우.  

### 마무리

어댑터 패턴(Adapter Pattern)은 기존 시스템과 새 시스템 간의 간극을 메우는 데 효과적인 방법을 제공한다.  
특히, 코드 변경 없이 새로운 요구사항에 유연하게 대처할 수 있는 강력한 도구로 활용된다.  

아래 글에서 다른 디자인 패턴들을 확인할 수 있다.  
[디자인 패턴 모음](https://gihak111.github.io/design_patterns/2024/11/05/Types_Of_Design_Patterns_upload.html)  
