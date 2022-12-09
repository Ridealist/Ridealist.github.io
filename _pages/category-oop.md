---
title: "OOP"
layout: archive
permalink: /oop/
author_profile: true
sidebar:
  nav: "sidebar-category"
---


{% assign posts = site.categories.oop %}
{% for post in posts %} {% include archive-single.html type=page.entries_layout %} {% endfor %}