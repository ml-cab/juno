# Juno Documentation

A 54-chapter reference, built as a [Jupyter Book](https://jupyterbook.org) (MyST Document
Engine) static site, presenting the Juno project's documentation as a single
cross-linked book organized by reader intent (getting started, architecture, guides, LoRA
fine-tuning, REST API, deployment, observability, testing, legal, project, releases).

## Structure

```
juno-documentation/
|-- myst.yml              # project config + table of contents (chapter order)
|-- index.md               # front matter, "how to read this book", full TOC
|-- references.md          # back matter: which docs/ source file each chapter came from
|-- part1/                 # Getting Started:                1.1-1.4
|-- part2/                 # Architecture:                   2.1-2.6
|-- part3/                 # CLI Reference:                  3.1-3.8
|-- part4/                 # LoRA Fine-Tuning:                4.1-4.8
|-- part5/                 # REST API:                        5.1-5.4
|-- part6/                 # Deployment:                      6.1-6.3
|-- part7/                 # Observability and Performance:   7.1-7.3
|-- part8/                 # Testing:                          8.1-8.2
|-- part9/                 # Legal and Compliance:             9.1-9.8
|-- part10/                # Community and Project:            10.1-10.6
|-- part11/                # Releases:                          11.1-11.2
`-- build.sh                # checks/installs the myst CLI and builds static HTML
```

Each chapter is its own Markdown file, named `NN-slug.md` within its part folder (for example
`part4/03-training-guide.md` is chapter 4.3). The file's local number within its part folder
matches the chapter's displayed decimal number.

## Cross-references (automatic, reorder-safe)

- Every chapter starts with a MyST label matching its decimal chapter number, for example
  `(ch-4-3)=` for chapter 4.3.
- In-text mentions of other chapters link to that label, for example `[Chapter 4.3](#ch-4-3)`.
  MyST resolves these project-wide, from any file, regardless of chapter order, so nothing
  breaks if you reorder or insert chapters.
- Each chapter ends with a "See also" list of related chapters and a Previous / Table of
  Contents / Next navigation footer.

## Inserting a new chapter

1. Add a new file inside the right part folder, for example `part7/04-my-new-topic.md`,
   starting with a unique label matching its intended decimal number:
   ```markdown
   (ch-7-4)=
   # 7.4. My New Topic
   ...
   ```
2. Add it to `myst.yml` under the right part's `children:` list, in the position you want.
3. Update the neighboring chapters' nav footers and `index.md`'s TOC to reference it by number.
4. Rebuild: `./build.sh`

Existing cross-references to other chapters keep working untouched; labels are not positional,
so nothing needs renumbering.

## Diagrams

Mermaid diagrams (fenced ` ```mermaid ` code blocks) are used throughout for architecture,
data-flow, and lifecycle diagrams, and render natively in the built site. Fixed-width ASCII art
is not used for structural or architectural diagrams in this book.

## Building

```bash
./build.sh              # build AND launch a working local preview automatically
./build.sh build-only   # just build _build/html, no server (for CI/deploy)
./build.sh serve        # live-reloading local preview
./build.sh clean        # remove build artifacts
```

The script checks for the `myst` CLI and installs it via `pip install mystmd` if missing (no
manual Node.js setup required in the common case).

## Publishing

`_build/html` is a complete static site; deploy it anywhere:

- **GitHub Pages**: `myst init --gh-pages` generates a ready-to-use GitHub Actions workflow.
- **Any static host** (Netlify, Cloudflare Pages, S3, nginx): upload the contents of
  `_build/html`.

No reader accounts, no paywalls, no server required.

## Relationship to the Juno source tree

The `juno-documentation` replaced the `docs` folder and holding most of the technical data.
The source files `CHANGELOG.md` `CLA.md` `CONTRIBUTORS.md` `FUNDING.md` `GOVERNANCE.md` `Legal.md` `RELEASE_NOTES.md` `SECURITY.md` are sharing the content with `juno-documentation` and have to be updated in sync!

The `README.md` of Juno is heavy referencing to `juno-documentation`