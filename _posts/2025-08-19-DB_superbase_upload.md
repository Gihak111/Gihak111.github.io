---
layout: single
title:  "Supabase로 모델을 저장한다"
categories: "AI"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## Supabase란
Supabase는 SQL 명령어로 쉽게 제어할 수 있는 오픈소스 데이터베이스 플랫폼이다.  
TensorFlow 모델 저장에 유용하며, 스토리지 설정과 권한 관리가 간편하다.  
이번 문서에서는 Supabase를 활용해 모델을 저장하는 방법을 설명한다.  

## 무료 티어 정보
Supabase 무료 티어는 다음과 같은 스펙을 제공한다:  
- **데이터베이스 저장 공간**: 500MB  
- **파일 스토리지**: 1GB  
- **대역폭**: 월 2GB  
- **제한**: API 요청은 분당 1,000회, 월간 활성 사용자(MAU)는 50,000명까지다.  

TensorFlow 모델 크기가 1GB를 초과하면 유료 플랜(월 $25부터)으로 전환해야 한다.  

## Supabase 스토리지 설정
Supabase에서 모델 파일을 저장하려면 스토리지를 설정하고 팀 단위로 권한을 관리해야 한다.  
아래는 `models` 버킷을 생성하고 팀 기반 RLS(Row-Level Security) 정책을 적용하는 과정이다.  
Supabase 대시보드의 **SQL Editor**에서 다음 코드를 순서대로 실행한다.  

### 1. 팀 테이블 생성
팀 정보를 저장하는 테이블을 생성한다.  
```sql
-- teams 테이블 생성
CREATE TABLE public.teams (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  name TEXT NOT NULL
);
```

### 2. 멤버 테이블 생성
팀과 사용자를 연결하는 테이블을 생성한다. 로그인한 사용자가 속한 팀을 관리한다.  
```sql
-- members 테이블 생성
CREATE TABLE public.members (
  team_id UUID REFERENCES public.teams(id) ON DELETE CASCADE NOT NULL,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  PRIMARY KEY (team_id, user_id)
);
```

### 3. 팀 목록 조회 함수
현재 로그인한 사용자가 속한 팀 ID를 반환하는 함수를 생성한다.  
```sql
-- 내가 속한 팀 목록을 가져오는 함수
CREATE OR REPLACE FUNCTION get_my_teams()
RETURNS SETOF uuid
LANGUAGE sql STABLE
AS $$
  SELECT team_id FROM public.members WHERE user_id = auth.uid();
$$;
```

### 4. 스토리지에 팀 ID 컬럼 추가
모델을 저장하는 `storage.objects` 테이블에 `team_id` 컬럼을 추가한다.  
```sql
-- storage.objects 테이블에 team_id 컬럼 추가
ALTER TABLE storage.objects
ADD COLUMN team_id UUID REFERENCES public.teams(id);
```

### 5. RLS 정책 적용
`models` 버킷에 팀 기반 접근 정책을 설정한다. 로그인한 사용자가 속한 팀의 모델에만 접근하도록 제한한다.  
```sql
-- Team Access Policy 생성
CREATE POLICY "Team Access Policy"
ON storage.objects
AS PERMISSIVE
FOR ALL
TO authenticated
USING (
  bucket_id = 'models'
  AND team_id IN (SELECT get_my_teams())
)
WITH CHECK (
  bucket_id = 'models'
  AND team_id IN (SELECT get_my_teams())
);
```

## 오류 해결 방법
위 과정을 따라도 정책 오류가 발생하면 기존 정책 충돌 가능성이 있다. 다음 단계를 따른다.  

### 1. 기존 정책 삭제
잘못된 정책을 제거한다. 오류가 발생해도 무시해도 된다(정책이 없던 경우일 수 있다).  
```sql
-- 기존 정책 삭제
DROP POLICY IF EXISTS "Allow ALL operations on models bucket" ON storage.objects;
```

### 2. 올바른 정책 생성
`models` 버킷에 대해 모든 작업을 허용하는 정책을 생성한다.  
```sql
-- 새 정책 생성
CREATE POLICY "Allow ALL operations on models bucket"
ON storage.objects
FOR ALL
TO authenticated
USING (
  bucket_id = 'models'
  AND team_id IN (SELECT get_my_teams())
)
WITH CHECK (
  bucket_id = 'models'
  AND team_id IN (SELECT get_my_teams())
);
```

### 3. 정책 확인
정책이 제대로 적용됐는지 확인한다.  
```sql
-- 정책 확인
SELECT * FROM pg_policies WHERE policyname = 'Allow ALL operations on models bucket';
```

정책 오류 없이 요청이 처리되면 성공이다.  

## TensorFlow 모델 업로드
Supabase 스토리지에 TensorFlow 모델을 업로드하려면 JavaScript SDK를 사용한다.  
먼저, Supabase 대시보드의 **Storage** 메뉴에서 `models` 버킷을 생성한다(버킷 이름은 SQL 코드의 `models`와 일치해야 한다).  
다음 코드를 참고해 모델 파일을 업로드한다:  
```javascript
import { createClient } from '@supabase/supabase-js';

// Supabase 클라이언트 초기화
const supabase = createClient('YOUR_SUPABASE_URL', 'YOUR_SUPABASE_KEY');

// 모델 파일 업로드
async function uploadModel(file, teamId) {
  const { data, error } = await supabase
    .storage
    .from('models')
    .upload(`models/${file.name}`, file, {
      metadata: { team_id: teamId }
    });
  
  if (error) {
    console.error('업로드 실패:', error);
    return;
  }
  console.log('업로드 성공:', data);
}

// 사용 예시
const modelFile = new File(['...'], 'model.h5'); // TensorFlow 모델 파일
uploadModel(modelFile, 'YOUR_TEAM_ID');
```

## 주의사항
- **버킷 이름**: SQL 코드와 업로드 코드의 버킷 이름(`models`)이 동일해야 한다.  
- **파일 크기**: 무료 티어는 1GB 스토리지 한도가 있으므로 모델 크기를 확인한다.  
- **메타데이터**: `team_id`를 메타데이터로 추가해야 RLS 정책이 작동한다.  
- **인증**: 업로드는 인증된 사용자(`authenticated`)로 로그인한 상태에서 가능하다. Supabase Authentication을 설정해야 한다.  

## 결론
Supabase로 TensorFlow 모델을 저장하는 방법을 정리했다.  
무료 티어는 1GB 스토리지와 월 2GB 대역폭을 제공하니 한도를 주의한다.  
위 단계를 따라 하면 정책 오류 없이 모델을 저장할 수 있다.  