---
layout: page
title: Projects
permalink: /projects/
hide_title: true
---
{%- comment -%} Data keys of papers shown on /publications/, to look them up by id {%- endcomment -%}
{%- assign listed_keys = "preprints,conferences,kci,demos" | split: "," -%}
{%- assign projects = site.data.projects | default: empty -%}

{%- if projects.size > 0 %}
  {%- for project in projects %}
<section class="project-card" id="{{ project.id }}">
  <h2>{{ project.title }}</h2>
  {%- if project.tagline %}
  <p class="project-tagline">{{ project.tagline }}</p>
  {%- endif %}
  {%- if project.image %}
  <figure class="project-figure">
    <img src="{{ project.image | relative_url }}" alt="{{ project.title }}" loading="lazy">
  </figure>
  {%- endif %}
  {%- if project.summary %}
  <div class="project-summary">{{ project.summary | markdownify }}</div>
  {%- endif %}
  {%- if project.video %}
  <figure class="project-video">
    {%- if project.video contains "youtu" %}
      {%- comment -%} YouTube watch, youtu.be or embed URL -> embed URL {%- endcomment -%}
      {%- assign video_id = project.video | split: "v=" | last | split: "youtu.be/" | last | split: "embed/" | last | split: "&" | first | split: "?" | first %}
    <iframe src="https://www.youtube-nocookie.com/embed/{{ video_id }}" title="{{ project.title }} demo video" loading="lazy" allow="encrypted-media; picture-in-picture; fullscreen" allowfullscreen></iframe>
    {%- else %}
    <video src="{{ project.video | relative_url }}" controls muted playsinline preload="metadata"></video>
    {%- endif %}
  </figure>
  {%- endif %}
  {%- include publication-links.html links=project.links %}
  {%- if project.login_required %}
    {%- assign contact_email = site.footer.contact | where: "name", "Email" | first %}
  <p class="project-access">
    Login required · Demo account available on request{% if contact_email %} via <a href="{{ contact_email.url }}?subject={{ project.title | append: ' demo account' | url_encode | replace: '+', '%20' }}">email</a>{% endif %}
  </p>
  {%- endif %}
  {%- if project.publications.size > 0 %}
  <h3 class="project-papers-heading">Related Papers</h3>
    {%- comment -%} Unlisted papers are not on /publications/, so they get a full row here, text only {%- endcomment -%}
    {%- for pub_id in project.publications %}
      {%- for item in site.data.publications.unlisted %}
        {%- if item.id == pub_id %}
    {% include publication.html publication=item anchor=false teaser=false %}
        {%- endif %}
      {%- endfor %}
    {%- endfor %}
    {%- comment -%} The rest link to their row on /publications/ {%- endcomment -%}
  <ul class="project-paper-links">
    {%- for pub_id in project.publications %}
      {%- for key in listed_keys %}
        {%- for item in site.data.publications[key] %}
          {%- if item.id == pub_id %}
    <li>
      <a href="{{ '/publications/' | relative_url }}#{{ item.id }}">{{ item.title }}</a>
      {%- if item.venue %}<span class="project-paper-venue">{{ item.venue }}</span>{% endif %}
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
<p class="pub-empty">Projects will be added soon.</p>
{%- endif %}
