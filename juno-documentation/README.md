# Juno Documentation

A 19-chapter reference, built as a [Jupyter Book](https://jupyterbook.org) (MyST Document
Engine) static site, restructuring the Juno project's `docs/` folder into a single cross-linked
book.

## Structure

```
juno-documentation/
├── myst.yml              # project config + table of contents (chapter order)
├── index.md              # front matter, "how to read this book", full TOC
├── references.md         # back matter - which original docs/ file each chapter came from
├── part1/                 # Getting Started:                chapters 1-7
├── part2/                 # LoRA Fine-Tuning:                chapters 8-10
├── part3/                 # Model Support and Performance:   chapters 11-13
├── part4/                 # Governance, Legal, and Compliance: chapters 14-19
├── assets/                 # images referenced by chapters (e.g. AWS deployment screenshot)
└── build.sh                # checks/installs the myst CLI and builds static HTML
```

Each chapter is its own Markdown file, named `NN-slug.md` (e.g.
`part1/02-architecture-reference.md`).

## Cross-references (automatic, reorder-safe)

- Every chapter starts with a MyST label: `(ch-02)=`
- Every in-text mention of "Chapter N" / "Chapters N-M" is a link to that label, e.g.
  `[Chapter 2](#ch-02)`. MyST resolves these project-wide, from any file, regardless of chapter
  order — so nothing breaks if you reorder or insert chapters.
- Each chapter ends with a Previous / Table of Contents / Next navigation footer.

## Inserting a new chapter

1. Add a new file, e.g. `part3/13b-my-new-topic.md`, starting with a unique label:
   ```markdown
   (ch-13b)=
   # 13b. My New Topic
   ...
   ```
2. Add it to `myst.yml` under the right part's `children:` list, in the position you want.
3. (Optional) Update the neighboring chapters' nav footers and `index.md`'s TOC if you want
   them to mention it by number.
4. Rebuild: `./build.sh`

Existing cross-references to other chapters keep working untouched — labels aren't positional,
so nothing needs renumbering.

## Diagrams

Mermaid diagrams (` ```mermaid ` fenced code blocks) are used throughout for architecture and
data-flow diagrams, and render natively in the built site.

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

`_build/html` is a complete static site — deploy it anywhere:

- **GitHub Pages**: `myst init --gh-pages` generates a ready-to-use GitHub Actions workflow.
- **Any static host** (Netlify, Cloudflare Pages, S3, nginx): upload the contents of
  `_build/html`.

No reader accounts, no paywalls, no server required.

## Relationship to the Juno source tree

This project is a documentation build only — it does not contain or depend on the Juno engine
source code. See [references.md](references.md) for a chapter-by-chapter map back to the
original files in Juno's `docs/` folder.
