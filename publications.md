---
layout: page
title: Publications
permalink: /publications/
hide_title: true
---
{%- assign section_keys = "preprints,conferences,demos" | split: "," -%}
{%- assign section_titles = "Preprints|Conference and Journal Papers|Posters, Demos, Workshop Papers" | split: "|" -%}
{%- comment -%} Every data key, including subsections, for the counts in the chips {%- endcomment -%}
{%- assign count_keys = "preprints,conferences,kci,demos" | split: "," -%}
{%- assign publication_count = 0 -%}
{%- for key in count_keys -%}
  {%- assign items = site.data.publications[key] | default: empty -%}
  {%- assign publication_count = publication_count | plus: items.size -%}
{%- endfor -%}

{%- if publication_count > 0 %}
<nav class="blog-categories" aria-label="Topics" data-pub-tags>
  <a class="blog-category-chip is-active" href="#" data-category="">All <span class="blog-category-count">{{ publication_count }}</span></a>
  {%- for tag in site.data.publication_tags %}
    {%- assign tag_count = 0 %}
    {%- for key in count_keys %}
      {%- for item in site.data.publications[key] %}{% if item.tags contains tag[0] %}{% assign tag_count = tag_count | plus: 1 %}{% endif %}{% endfor %}
    {%- endfor %}
    {%- if tag_count > 0 %}
  <a class="blog-category-chip" href="#{{ tag[0] }}" data-category="{{ tag[0] }}">{{ tag[1] }} <span class="blog-category-count">{{ tag_count }}</span></a>
    {%- endif %}
  {%- endfor %}
</nav>
  {%- for key in section_keys %}
    {%- assign items = site.data.publications[key] | default: empty %}
    {%- assign sub_items = empty %}
    {%- if key == "conferences" %}{% assign sub_items = site.data.publications.kci | default: empty %}{% endif %}
    {%- if items.size > 0 or sub_items.size > 0 %}
<section class="pub-section" data-filter-group>
  <h2>{{ section_titles[forloop.index0] }}</h2>
  {%- for item in items %}
    {% include publication.html publication=item %}
  {%- endfor %}
  {%- if sub_items.size > 0 %}
  <div class="pub-subsection" data-filter-group>
    <h3>KCI (Korea Citation Index)</h3>
    {%- for item in sub_items %}
      {% include publication.html publication=item %}
    {%- endfor %}
  </div>
  {%- endif %}
</section>
    {%- endif %}
  {%- endfor %}
{%- else %}
<h2>Publications</h2>
<p class="pub-empty">Publications will be added as they become available.</p>
{%- endif %}
