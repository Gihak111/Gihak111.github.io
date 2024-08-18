---
layout: single
title:  "jekyll 블로그와 깃허브 repository 연동"
categories: "jekyll"
tag: "code"
toc: true
author_profile: false
sidebar:
    nav: "docs"
---

# jekytll
자신의 깃허브에 올라가 잇는 repository는 github API를 통해서 플로그에 링크를 띄울 수 있다.  
다음의 코들르 참고하자.  
```html
---
layout: default
title: Projects
---

<div class="container">
    <h1>Projects</h1>
    <div class="project-list">
        <ul id="repo-list">
            <!-- Repositories will be loaded here -->
        </ul>
    </div>
</div>

<style>
    #repo-list {
        font-size: 1.2em; /* 글자 크기를 원하는 만큼 조정할 수 있다. */
    }

    #repo-list a {
        font-size: 1.2em; /* 링크의 글자 크기도 조정할 수 있다. */
    }
</style>

<script>
    async function fetchRepos() {
        const username = 'Gihak111';  //내 깃허브 이름
        const apiUrl = `https://api.github.com/users/${username}/repos?type=public`;

        try {
            const response = await fetch(apiUrl);
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            const repos = await response.json();
            const repoList = document.getElementById('repo-list');
            repos.forEach(repo => {
                const listItem = document.createElement('li');
                const link = document.createElement('a');
                link.href = repo.html_url;
                link.textContent = repo.name;
                listItem.appendChild(link);
                repoList.appendChild(listItem);
            });
        } catch (error) {
            console.error('Error fetching repositories:', error);
            document.getElementById('error-message').textContent = 'Failed to load repositories. Please try again later.';
        }
    }

    document.addEventListener('DOMContentLoaded', fetchRepos);
</script>

```
위 코드를 보면, 깃허브 API 링크를 통해서 간단하게 포스트를 가져오는 것을 볼 수 있다.  