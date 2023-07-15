---
title: "HCI"
layout: archive
permalink: /hci/
author_profile: true
sidebar:
  nav: "sidebar-category"
---

{% assign posts = site.categories.hci %}
{% for post in posts %} {% include archive-single.html type=page.entries_layout %} {% endfor %}