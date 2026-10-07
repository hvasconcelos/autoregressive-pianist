# The Autoregressive Pianist: book sources

The Markdown files here are the source of truth for the ebook. Edit them, then rebuild.

```
docs/
  book.yaml      title, subtitle, date, language, cover, stable EPUB identifier
  epub.css       the book's stylesheet
  images/        cover and figures (referenced from chapters as ../images/...)
  chapters/      one Markdown file per chapter, part divider or appendix
  build.py       builds the EPUB with pandoc
  build/         output (git-ignored)
```

## Build

Needs [pandoc](https://pandoc.org) 3 (`brew install pandoc`).

```bash
uv run python docs/build.py                  # docs/build/The-Autoregressive-Pianist.epub
uv run python docs/build.py --format html    # one self-contained HTML page for proofreading
```

## Writing conventions

- **Order**: chapters are built in file-name order, so the two-digit prefix sets the reading order. To insert a chapter, give it a prefix that sorts where you want it, or renumber.
- **One `#` heading per file.** Each top-level heading starts a new file in the EPUB and a new entry in the contents. `##` headings are the sections listed under it. Chapter and section numbers are written by hand (`# 7. Tags…`, `## 7.1 …`).
- **Part dividers** use `# Part I: The small model {.part}`.
- **Code listings** are fenced blocks with a language (` ``` python `, ` ``` bash `, ` ``` text `). A file name label goes just above the listing:

  ```markdown
  ::: filename
  tags.py
  :::
  ```

- **Notes and warnings**: the first paragraph is the title.

  ```markdown
  ::: {.admonition .note}
  Why there is no tempo tag

  Tempo means beats per minute…
  :::
  ```

  Use `{.admonition .warning}` for the amber variant.
- **Figures** are an image alone in its paragraph; the alt text becomes the caption: `![**Figure 1.** The five stages…](../images/pipeline.png)`. Figure numbers are written by hand.
- **Quotes** stay straight: the build turns off smart quotes, so `"` and `'` appear exactly as typed.
