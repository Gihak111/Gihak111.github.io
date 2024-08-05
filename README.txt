jekyll new --skip-bundle
jekyll new docs --skip-bundle

https://pages.github.com/versions/
gem "github-pages", "~> 231", group: :jekyll_plugins
$ git checkout --orphan gh-pages
$ git rm -rf .
bundle install
bundle exec jekyll server
http://127.0.0.1:4000/

사진 만들기
https://www.bing.com/images/create?FORM=GDPGLP

젤키 테마 새로 만들 떄
