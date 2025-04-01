
### 깃허브 블로그용 벡엔드 입니다.  
jekyll 기반으로 만들었습니다.  
직접 만들었고, 여러가지 테마의 백엔드를 뜯어보면서,  
배운 내용들을 종합해서 만들었습니다.  
블로그 테마는, 다음 레포지토리에서 다운받을 수 있습니다.  

---
### 새로운 블로그 태마 만들기
```bash
jekyll new --skip-bundle
jekyll new docs --skip-bundle
```
위 코드를 통해서 텅 빈 제킬 블로그 테마를 만들 수 있다.  

다음 링크에 가서, 버전을 확인하자.  
[링크](https://pages.github.com/versions/)

```bash
gem "github-pages", "~> 231", group: :jekyll_plugins
$ git checkout --orphan gh-pages
$ git rm -rf .
bundle install
bundle exec jekyll server
http://127.0.0.1:4000/
```
위와 같은 방법으로 로컬 서버를 열어서 확인 할 수 있다.  
블로그 내에서 발생하는 오류들을 로그를 통해 잡을 수 있으며,  
백엔드에 변동사항을 줄 시 즉각적으로 로컬 서버에 적용되므로 커밋전에 확인하기 좋다.  