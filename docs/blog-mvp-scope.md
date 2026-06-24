# Blog MVP Scope and Preservation Policy

This document defines the first-phase migration scope for the blog redesign based on the Jekyll reference site at `reference/yoonsu0816.github.io`.

## Goal

Redesign the blog homepage with a Minimal Light-inspired personal homepage layout while preserving the existing Jekyll blog structure, post URLs, category URLs, comments, analytics, and deployment flow.

The first MVP is intentionally limited to the work covered by GitHub issues #6-#11 plus the verification work in #12.

## First MVP Scope

The first MVP includes:

- Defining and documenting the migration scope and preservation policy.
- Preparing profile and site metadata for the new homepage.
- Adding a homepage-only layout and homepage-only styles.
- Rebuilding `/` as a profile-centered homepage.
- Adding homepage sections for recent posts and category navigation.
- Keeping navigation between the new homepage and existing blog pages intact.
- Verifying Jekyll build, responsive behavior, and GitHub Pages compatibility.

The first MVP does not include redesigning post detail pages or category archive pages. Those pages may be visually tuned in a later phase after the homepage migration is stable.

## Preservation Policy

The following must be preserved during the first MVP:

- Existing `_posts` content. At planning time, the repo contains 53 Markdown posts.
- Existing permalink policy: `/:categories/:title/`.
- Existing category pages under `_pages/category-*.md`.
- Existing category routes such as `/llm/`, `/hci/`, `/woowacourse/`, and related archive URLs.
- Existing utterances comment setup.
- Existing Google Analytics setup.
- Existing favicon and logo behavior unless a later issue explicitly changes the asset.
- Existing Jekyll and GitHub Pages deployment model.

Implementations should avoid changing post front matter, post filenames, category names, or permalink-related configuration unless a later issue explicitly scopes that work.

## Change Policy

The following changes are allowed in the first MVP:

- Add a new homepage layout, expected to be `_layouts/homepage.html`.
- Add homepage-only styles, scoped so they do not leak into existing post or category pages.
- Convert the root page `/` from the current Minimal Mistakes home/archive layout into the new homepage layout.
- Use existing Jekyll data, especially `_config.yml` and `_data/navigation.yml`, to populate the homepage where practical.
- Add small includes for homepage recent posts and category links.
- Add only the reference assets that are necessary for the MVP.

Homepage styles should be scoped under a wrapper class such as `.homepage-wrapper` or an equivalent layout-specific root. Reference CSS must not be copied globally without scoping because the reference theme uses broad selectors such as `header`, `section`, `h1`, and `a`.

## Excluded Work

The following are explicitly out of scope for the first MVP:

- Replacing the current theme with `remote_theme: yaoyao-liu/minimal-light`.
- Migrating the site to Next.js.
- Migrating existing posts to a new content model.
- Redesigning all post detail pages.
- Redesigning all category and archive pages.
- Changing the permalink structure.
- Changing the comment provider.
- Copying the entire reference repository or all reference assets into the production site.

## Follow-up Issue Mapping

- #7: Normalize profile and site metadata for the new homepage.
- #8: Add the homepage-only layout and scoped styles.
- #9: Rebuild homepage content around profile and introduction sections.
- #10: Add homepage recent-post and category-link components.
- #11: Resolve navigation and static asset behavior between the new homepage and existing pages.
- #12: Verify build, responsive behavior, and deployment compatibility.

## Completion Criteria for #6

Issue #6 is complete when:

- This scope and preservation policy is committed to the repo.
- The first MVP scope is clearly limited to #6-#11 plus #12 verification.
- Preserved, changed, and excluded items are explicit.
- Existing blog URL preservation is documented as a hard requirement.
