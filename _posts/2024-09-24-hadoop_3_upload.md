---
layout: single
title:  "하둡 강좌 3편 MapReduce"
categories: "hadoop"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

### 하둡 강좌 3편: **MapReduce 개념과 기본 사용법**  
이번 강좌에서는 하둡의 핵심 컴포넌트 중 하나인 **MapReduce**에 대해 다뤄보겠다.  
MapReduce는 대규모 데이터를 처리하기 위한 프로그래밍 모델로, 데이터를 분산하여 처리하고 병합하는 과정을 효율적으로 수행할 수 있다.  
이번 강의에서는 MapReduce의 기본 개념과 구조를 살펴본 후, 간단한 예제를 통해 실습해보자.  

## 1. MapReduce란?  
MapReduce는 크게 두 가지 작업으로 나뉜다:  
- **Map**: 입력 데이터를 키-값 쌍으로 변환하고, 각 키를 기준으로 데이터를 분할한다.  
- **Reduce**: Map 작업의 결과를 다시 병합하여 최종 출력을 생성한다.  
하둡에서 MapReduce는 분산된 데이터를 처리하는 데 최적화되어 있으며, 각 작업은 클러스터 내 여러 노드에서 병렬로 실행된다.  
이 과정을 통해 대용량 데이터도 빠르게 처리할 수 있다.  

## 2. MapReduce 처리 흐름  
MapReduce의 처리 과정은 다음과 같이 요약할 수 있다:  
1. **입력 데이터**: HDFS에서 데이터를 읽어와 처리할 준비를 한다.  
2. **Map 작업**: 입력 데이터를 여러 조각으로 나누고, 각 조각을 키-값 쌍으로 변환한다.  
3. **Shuffle 및 Sort**: 같은 키를 가진 데이터를 모아 정렬한다.  
4. **Reduce 작업**: 같은 키에 대한 데이터를 모아 최종 결과를 생성한다.  
5. **출력 데이터**: 결과를 HDFS에 저장한다.  

### 예시: 단어 빈도수 세기 (Word Count)  
가장 기본적인 MapReduce 예제는 단어 빈도수를 세는 작업이다.  
주어진 텍스트에서 각 단어가 몇 번 등장하는지 계산하는 것이다.  
다음 예제를 통해 이 과정을 구현해보자.  

## 3. MapReduce 프로그램 작성  
### 3.1 Mapper 클래스  
먼저 **Mapper** 클래스를 정의해보자. Mapper는 입력 데이터를 처리하고, 각 단어를 키로, 값은 `1`로 매핑한다.  
```java
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

public class WordCountMapper extends Mapper<LongWritable, Text, Text, IntWritable> {

    private final static IntWritable one = new IntWritable(1);
    private Text word = new Text();

    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String line = value.toString();
        String[] words = line.split("\\s+");  // 공백을 기준으로 단어를 나눈다.

        for (String w : words) {
            word.set(w);
            context.write(word, one);  // (단어, 1) 형태로 출력
        }
    }
}
```  

### 3.2 Reducer 클래스  
**Reducer** 클래스는 Mapper에서 생성된 `(단어, 1)` 형태의 쌍을 받아, 각 단어의 빈도수를 계산한다.  
```java
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

public class WordCountReducer extends Reducer<Text, IntWritable, Text, IntWritable> {

    @Override
    protected void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
        int sum = 0;
        for (IntWritable val : values) {
            sum += val.get();  // 각 단어의 값(1)을 합산한다.
        }
        context.write(key, new IntWritable(sum));  // (단어, 빈도수) 형태로 출력
    }
}
```  

### 3.3 Driver 클래스  
마지막으로 **Driver** 클래스는 MapReduce 작업을 설정하고 실행하는 역할을 한다.  
```java
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class WordCountDriver {

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "word count");

        job.setJarByClass(WordCountDriver.class);
        job.setMapperClass(WordCountMapper.class);
        job.setReducerClass(WordCountReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));  // 입력 경로 설정
        FileOutputFormat.setOutputPath(job, new Path(args[1]));  // 출력 경로 설정

        System.exit(job.waitForCompletion(true) ? 0 : 1);  // 작업 완료 여부에 따라 종료
    }
}
```  

## 4. MapReduce 실행  
### 4.1 컴파일 및 패키징  
Hadoop MapReduce 프로그램은 Java로 작성되므로, 컴파일한 후 JAR 파일로 패키징해야 한다.  
다음 명령어를 통해 Maven을 사용하여 빌드하자.  
```bash
mvn clean package
```  
이 명령어는 Maven 프로젝트를 빌드하고, `target/` 디렉토리에 JAR 파일을 생성한다.  

### 4.2 HDFS에 데이터 업로드  
MapReduce를 실행하기 전에, 처리할 입력 데이터를 HDFS에 업로드해야 한다. `input.txt`라는 파일을 HDFS에 업로드하자.  
```bash
hdfs dfs -put input.txt /user/hadoop/wordcount/input
```  

### 4.3 MapReduce 작업 실행  
이제 MapReduce 작업을 실행할 차례이다. 다음 명령어를 통해 WordCount 작업을 실행한다.  
```bash
hadoop jar target/wordcount-1.0.jar WordCountDriver /user/hadoop/wordcount/input /user/hadoop/wordcount/output
```  
- `/user/hadoop/wordcount/input`은 입력 데이터가 있는 경로이고,  
- `/user/hadoop/wordcount/output`은 출력 데이터가 저장될 경로이다.  

### 4.4 결과 확인  
MapReduce 작업이 완료되면, 출력 경로에 결과가 저장된다.  
결과 파일을 확인해보자.  
```bash
hdfs dfs -cat /user/hadoop/wordcount/output/part-r-00000
```  

이 명령어는 MapReduce 작업의 결과를 출력한다.  

## 5. 마무리  
이번 강좌에서는 MapReduce의 개념을 배우고, 간단한 WordCount 예제를 통해 MapReduce 작업을 실행하는 방법을 익혔다.  
MapReduce는 하둡의 핵심 컴포넌트 중 하나로, 대규모 데이터를 병렬로 처리할 수 있는 강력한 도구이다.  
앞으로의 강좌에서는 더 복잡한 MapReduce 작업을 다루고, 다양한 실습을 통해 이를 익히는 시간을 가져보겠다.  

---

다음 강좌에서는 **하둡 에코시스템**의 다른 중요한 컴포넌트들, 예를 들어 **YARN**과 **Hive** 등을 다뤄보겠다.  