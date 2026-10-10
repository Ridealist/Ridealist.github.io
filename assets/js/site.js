(function () {
  // Open links to other sites in a new tab
  document.querySelectorAll("a[href]").forEach(function (link) {
    if (/^https?:$/.test(link.protocol) && link.host !== location.host) {
      link.target = "_blank";
      link.rel = "noopener";
    }
  });

  // Mobile menu toggle
  var button = document.querySelector(".site-nav .menu-icon");
  var menu = document.querySelector(".site-nav .menu");
  if (button && menu) {
    button.addEventListener("click", function () {
      var open = menu.classList.toggle("is-open");
      button.setAttribute("aria-expanded", open ? "true" : "false");
    });
  }

  // Table of contents for posts with `toc: true`
  var toc = document.querySelector("[data-toc]");
  var content = document.querySelector(".post-content");
  if (toc && content) {
    var list = toc.querySelector("ul");
    var headings = content.querySelectorAll("h2[id], h3[id]");
    headings.forEach(function (heading) {
      var item = document.createElement("li");
      item.className = "toc-" + heading.tagName.toLowerCase();
      var link = document.createElement("a");
      link.href = "#" + heading.id;
      link.textContent = heading.textContent;
      item.appendChild(link);
      list.appendChild(item);
    });
    if (headings.length > 0) toc.hidden = false;
  }
  // Home news: "Show more" reveals `data-step` more items per click; "Show less" folds back
  var news = document.querySelector("[data-news]");
  var newsActions = news && news.querySelector(".news-actions");
  if (newsActions) {
    var step = parseInt(newsActions.getAttribute("data-step"), 10);
    var moreButton = newsActions.querySelector("[data-news-more]");
    var lessButton = newsActions.querySelector("[data-news-less]");
    var newsSep = newsActions.querySelector(".news-sep");
    var extras = Array.prototype.slice.call(news.querySelectorAll("[data-news-extra]"));
    var updateNews = function () {
      var remaining = extras.filter(function (row) { return row.hidden; }).length;
      var expanded = remaining < extras.length;
      moreButton.hidden = remaining === 0;
      moreButton.textContent = "Show " + Math.min(step, remaining) + " more ↓";
      lessButton.hidden = !expanded;
      newsSep.hidden = moreButton.hidden || lessButton.hidden;
    };
    moreButton.addEventListener("click", function () {
      extras.filter(function (row) { return row.hidden; }).slice(0, step).forEach(function (row) {
        row.hidden = false;
      });
      updateNews();
    });
    lessButton.addEventListener("click", function () {
      extras.forEach(function (row) { row.hidden = true; });
      updateNews();
      moreButton.focus({ preventScroll: true });
      // The list just got shorter; bring the News heading back into view if it scrolled off
      var heading = news.previousElementSibling || news;
      if (heading.getBoundingClientRect().top < 0) heading.scrollIntoView({ behavior: "smooth" });
    });
  }

  // Chip filters driven by the URL hash: blog categories (/blog/#llm) and publication topics (/publications/#nlp)
  // With `limit`, the unfiltered list shows only the first `limit` items until the `more` button is clicked.
  var setupFilter = function (chips, groupSelector, limit, more) {
    if (!chips) return;
    var expanded = false;
    var applyFilter = function () {
      var current = decodeURIComponent(location.hash.slice(1));
      if (!chips.querySelector('[data-category="' + current + '"]')) current = "";
      chips.querySelectorAll("[data-category]").forEach(function (chip) {
        var active = chip.getAttribute("data-category") === current;
        chip.classList.toggle("is-active", active);
        chip.setAttribute("aria-current", active ? "true" : "false");
      });
      if (current) expanded = false; // back on All, the list folds again
      var max = !current && limit && !expanded ? limit : Infinity;
      var shown = 0;
      var total = 0;
      document.querySelectorAll(groupSelector).forEach(function (group) {
        var visible = 0;
        group.querySelectorAll("[data-categories]").forEach(function (item) {
          var match = !current || item.getAttribute("data-categories").split(" ").indexOf(current) !== -1;
          if (match) total++;
          item.hidden = !match || shown >= max;
          if (!item.hidden) { shown++; visible++; }
        });
        group.hidden = visible === 0;
      });
      if (more) more.hidden = shown >= total;
    };
    if (more) more.querySelector("button").addEventListener("click", function () {
      expanded = true;
      applyFilter();
    });
    chips.querySelector('[data-category=""]').addEventListener("click", function (event) {
      event.preventDefault();
      history.pushState(null, "", location.pathname);
      applyFilter();
    });
    window.addEventListener("hashchange", applyFilter);
    applyFilter();
  };
  setupFilter(document.querySelector("[data-blog-categories]"), "[data-blog-year]", 10, document.querySelector("[data-show-more]"));
  setupFilter(document.querySelector("[data-pub-tags]"), "[data-filter-group]");
})();
