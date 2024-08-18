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



project 변경 전
---
layout: default
title: Projects
---

<div class="container">
    <h1>Projects</h1>
    <div class="project-list">
        {% assign project_posts = site.posts | where: "tags", "projects" %}
        <ul>
            {% for post in project_posts %}
            <li><a href="{{ post.url }}">{{ post.title }}</a></li>
            {% endfor %}
        </ul>
    </div>
</div>
