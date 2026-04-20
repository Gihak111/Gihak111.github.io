---
layout: single
title: "Gemma4"
categories: "AI"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

## Vecel
우리가 Next 프로젝트를 하나 성공했다 치자.  
그럼 배포를 해야 할 거 아니냐  
근데 어? 배포 어캐하는거지? 플렛폼에 올려야 하나? 어캐하지? 싶은텐데,  
그냥 깃에 올려버리고 Vecel로 배포하면 편하다.  
깃 프리베이트여도 잘 배포되고, .env도 따로 관리할 수 있으니까 진짜 좋다.  
이거 어캐하는건지 한번 해 보자.  

## 1. 프로젝트 다이어트 및 깃허브 업로드
일단 프로젝트 용량부터 줄여야 한다. 로컬에 있는 모듈 포함 업로드는 깃허브 기본 용량인 100m를 초과하기에 제외하고 업로드 해야 한다.  

1. `.gitignore` 세팅  
   * 프로젝트 맨 위에 `.gitignore` 파일을 만들고, 깃허브에 절대 올라가면 안 되는 .env, node_modul, next 폴더 적어준다.  
   * `node_modules/` (외부 패키지 - 용량 괴물)  
   * `.next/` (로컬 빌드 결과물 - 얘도 용량 괴물)  
   * `.env` & `.env.local` (보안 키랑 DB 주소 - 털리면 큰일남)  
2. 잘못 올라간 파일 빼기  
   * 이미 깃허브에 무거운 폴더가 올라가 버렸으면 내 컴퓨터에서는 안 지우고 깃(Git) 감시망에서만 쏙 빼내는 명령어가 있다.  
   * `git rm -r --cached node_modules`  
   * `git rm -r --cached .next`  
3. 가벼워진 상태로 Push  
   * 이제 `git add .`, `git commit -m "..."`, `git push` 하면 수 MB 수준으로 깔끔하게 깃허브에 올라간다.  


## 2. Vercel 초기 배포 및 TypeScript 에러 잡기  
자, 가벼워진 레포지토리를 Vercel에 연결해서 빌드를 해보자. 근데 에러가 뜬다.  

1. Vercel 프로젝트 가져오기  
   * Vercel 대시보드 가서 `Add New > Project` 누르고 방금 올린 깃허브 레포지토리 가져오면 끝이다.   
2. 빌드 실패 (타입/린트 에러) 해결  
   * Vercel 화면에서는 에러가 잘려서 잘 안 보일 수 있다. 그럴 땐 내 컴퓨터에서 `npm run build` 쳐보면 진짜 원인이 나온다.  
   * `page.tsx` 파일에 쓸데없이 달려있던 `// @ts-expect-error` 주석 때문에 꼬인 거면 그냥 지워버리자.  
   * `next.config.mjs`에 옛날 문법(`serverExternalPackages`) 쓴 게 있으면 `experimental: { serverComponentsExternalPackages }`로 깔끔하게 바꿔주면 해결된다.  


## 3. 환경 변수(env) 및 인증(Auth) 세팅
빌드 성공해서 신나게 들어갔는데 `500 에러(Invalid URL)` 뜨면 화가난다.  
서버가 비밀번호를 몰라서 그러는 거니까 세팅해 주자.  

1. 보안 키 생성 및 등록  
   * 터미널 열고 Node.js 명령어로 32바이트짜리 튼튼한 무작위 키(`AUTH_SECRET`) 하나 뽑아준다.  
   * 그다음 Vercel 대시보드 `Settings > Environment Variables` 가서 이 키를 등록하면 된다.  
2. 기준 주소(URL) 알려주기  
   * NextAuth가 길 잃어버리지 않게 `NEXTAUTH_URL` (또는 `AUTH_URL`) 변수에 Vercel 배포 주소(`https://erp-...vercel.app`) 딱 넣어주자.  
3. 쿠키 충돌 해결  
   * 새 암호화 키 적용했더니 예전 쿠키랑 싸워서 로그인이 안 될 때가 있다.  
   * 이건 걍 브라우저 개발자 도구 열어서 예전 쿠키 싹 날리거나, 시크릿 창 열어서 테스트해보면 깔끔하게 해결된다.  


## 4. 대망의 DB 클라우드 이사
여기가 제일 중요하다! 로컬에서 쓰던 `dev.db` (SQLite) 파일, 이거 Vercel에 올리면 매번 리셋돼서 날아간다.  
때문에 개발이 끝나면 프리즈마 같은 진짜 디비로 가야 한다.  
이것도 개꿀인게, Vecel에 프리즈마나 몽고디비 같은거도 다 있어서 쉽게 연동할 수 있다.  

1. Vercel Postgres 만들고 연결  
   * Vercel 대시보드 Storage 탭에서 무료 Postgres DB 하나 파고, 내 프로젝트에 `Connect` 누르면 된다. (혹시 기존에 수동으로 넣었던 `DATABASE_URL` 있으면 충돌 나니까 지우고 다시 깔끔하게 연결하자.)  
2. 내 컴퓨터로 환경 변수 땡겨오기  
   * 내 컴퓨터에서도 클라우드 DB에 접근해야 작업할 거 아니냐. 터미널에서 연결 좀 해주자.  
   * `npx vercel link` (프로젝트 연결)  
   * `npx vercel env pull .env.local` (Vercel에 있는 DB 주소들 내 컴퓨터로 쏙 뽑아오기)  
3. Prisma 설정 바꾸기  
   * `.env.local`에 들어온 변수들을 Prisma가 볼 수 있게 `.env` 파일에 그대로 복사해 주자.  
   * 그다음 `schema.prisma` 파일 열어서 DB 종류를 `sqlite`에서 `postgresql`로 싹 바꾸고, `url`이랑 `directUrl`에 변수 이름 딱 맞춰준다.  
4. 새 DB 뼈대 세우기 (마이그레이션)  
   * 로컬에 있던 옛날 `dev.db`랑 `prisma/migrations` 폴더는 쿨하게 지워버리자.  
   * 이제 `npx prisma migrate dev --name init` 딱 쳐주면 새 클라우드 DB에 첫 테이블 뼈대가 예쁘게 세워진다!  


## 5. 완벽한 자동 배포 파이프라인 구축
거의 다 왔다. 이제 깃허브에 코드 올릴 때마다 Vercel이 알아서 DB 구조까지 업데이트하게 만들어보자.  

1. `package.json` 스크립트 수정  
   * `"build"` 명령어를 `"prisma migrate deploy && next build"`로 바꿔준다.  
   * 이렇게 하면 Vercel이 배포할 때마다 알아서 DB 변경점 체크해서 안전하게 반영(`migrate deploy`)하고, 그다음에 Next.js를 빌드한다.  
   * *(참고로 Vercel 운영 서버에서는 `db push` 절대 쓰지 말고 `migrate deploy` 쓰는 게 국룰이다!)*  
2. 최종 Push 그리고 완성  
   * 바뀐 스크립트랑 마이그레이션 폴더를 깃허브에 싹 다 Push 하자.  
   * Vercel 화면에 초록색 체크마크 뜨면서 배포 성공하고, 클라우드 DB에 테스트 계정 들어가서 실제 온라인상에서 로그인까지 잘 되면 끝난 거다.  


## 결론
끝.  
플렛폼이지 사실.  
근데 막 이것저것 많지 않냐 나도 많이 써 봤는데 제일 편하게 한 것 같아서 좋았다.  