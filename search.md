---
layout: null
permalink: /search.md
---
[
  {% for post in site.posts %}
    {
      "title": "{{ post.title | escape }}",
      "url": "{{ site.baseurl }}{{ post.url }}"
    }{% if forloop.last %}{% else %},{% endif %}
  {% endfor %}
]
