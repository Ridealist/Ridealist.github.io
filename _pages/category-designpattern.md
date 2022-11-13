---
title: "DesignPattern"
layout: archive
permalink: /designpattern/
author_profile: true
sidebar:
  nav: "sidebar-category"
---


{% assign posts = site.categories.designpattern %}
{% for post in posts %} {% include archive-single.html type=page.entries_layout %} {% endfor %}