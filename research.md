---
layout: page
title: Research
permalink: /research/
hide_title: true
---
{%- comment -%} Data keys of papers shown on /publications/, to look them up by id {%- endcomment -%}
{%- assign listed_keys = "preprints,conferences,kci,demos" | split: "," -%}
{%- assign projects = site.data.research | default: empty -%}

{%- if projects.size > 0 %}
  {%- for project in projects %}
<section class="research-project" id="{{ project.id }}">
  <h2>{{ project.title }}</h2>
  {%- if project.image %}
  <figure class="research-figure">
    <img src="{{ project.image | relative_url }}" alt="{{ project.title }}" loading="lazy">
  </figure>
  {%- endif %}
  {%- if project.summary %}
  <div class="research-summary">{{ project.summary | markdownify }}</div>
  {%- endif %}
  {%- include publication-links.html links=project.links %}
  {%- if project.publications.size > 0 %}
  <h3 class="research-papers-heading">Related Papers</h3>
    {%- comment -%} Unlisted papers are not on /publications/, so they get a full row here, text only {%- endcomment -%}
    {%- for pub_id in project.publications %}
      {%- for item in site.data.publications.unlisted %}
        {%- if item.id == pub_id %}
    {% include publication.html publication=item anchor=false teaser=false %}
        {%- endif %}
      {%- endfor %}
    {%- endfor %}
    {%- comment -%} The rest link to their row on /publications/ {%- endcomment -%}
  <ul class="research-paper-links">
    {%- for pub_id in project.publications %}
      {%- for key in listed_keys %}
        {%- for item in site.data.publications[key] %}
          {%- if item.id == pub_id %}
    <li>
      <a href="{{ '/publications/' | relative_url }}#{{ item.id }}">{{ item.title }}</a>
      {%- if item.venue %}<span class="research-paper-venue">{{ item.venue }}</span>{% endif %}
    </li>
          {%- endif %}
        {%- endfor %}
      {%- endfor %}
    {%- endfor %}
  </ul>
  {%- endif %}
</section>
  {%- endfor %}
{%- else %}
<p class="pub-empty">Research projects will be added soon.</p>
{%- endif %}
