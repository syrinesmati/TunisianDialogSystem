# PFA_Report — End of Year Project

This repository contains the LaTeX source for the project report: *Towards Dialogue Systems for Low-Resource Languages*.

This README explains, step by step, how to set up a development environment from scratch (Windows-focused), build the PDF, and collaborate via Git.

## Prerequisites (Windows)

- VS Code (recommended) — https://code.visualstudio.com/
- MiKTeX (or TeX Live). Recommended: MiKTeX (winget install MiKTeX.MiKTeX).
- Perl (required by `latexmk`) — Strawberry Perl (optional if you use xelatex/bibtex sequence).
- (Optional) `pdftotext` / Poppler utilities if you need to extract PDF text.
- Git and a Git remote (e.g., GitHub, GitLab): https://git-scm.com/
- VS Code extension: LaTeX Workshop (to build and preview from the editor).

Notes:
- If you install MiKTeX via winget, the binaries will typically be under `%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64`.
- If `latexmk` fails complaining about `perl` not found, install Strawberry Perl or use the manual build sequence below.

## Quick build commands

Open a terminal in the repository root (where `main.tex` lives) and run:

PowerShell / CMD (manual sequence, robust):
```powershell
xelatex -synctex=1 -interaction=nonstopmode -file-line-error main.tex
bibtex main
xelatex -interaction=nonstopmode -file-line-error main.tex
xelatex -interaction=nonstopmode -file-line-error main.tex
```

Notes:
- The first `xelatex` pass writes auxiliary files (`.aux`, `.lof`, `.lot`, `.toc`).
- `bibtex` produces `main.bbl` from `references.bib`.
- Two more `xelatex` runs are necessary to finalise cross-references and lists.
- If you have `latexmk` and `perl` installed, you can instead run:
```powershell
latexmk -pdf -xelatex -interaction=nonstopmode -file-line-error main.tex
```

## VS Code setup

1. Open the repository folder in VS Code.
2. Install the `LaTeX Workshop` extension.
3. The repository includes workspace settings at `.vscode/settings.json` that define a recipe using `xelatex` + `bibtex` (configured to the local MiKTeX executables). If your installation paths differ, open `.vscode/settings.json` and update the `command` fields to point to your xelatex/bibtex executables.
4. Use the LaTeX Workshop build command from the Command Palette (`Ctrl+Shift+P` → `LaTeX Workshop: Build with recipe`).

## Fonts and Arabic support

This document uses Arabic text and polyglossia. The template sets the Arabic font in the source. If compilation complains about missing font metrics (e.g., `Amiri`), either install the requested font or modify the font selection in `main.tex` near the top. Example (works when `Arial Unicode MS` or another Arabic-capable font is installed):

```tex
\newfontfamily\arabicfont[Script=Arabic]{Arial Unicode MS}
```

To install `Amiri`, download and install it from https://www.amirifont.org/ or the CTAN package.

## Troubleshooting

- Empty Table of Contents / List of Figures / List of Tables: run the build sequence above (xelatex → bibtex → xelatex → xelatex). These lists require multiple runs to populate.
- `latexmk` reports missing `perl`: install Strawberry Perl (https://strawberryperl.com/) or use the manual sequence above.
- Missing Arabic glyphs / TFM errors: ensure the Arabic font is installed and available to XeLaTeX.
- If VS Code's PDF viewer shows an old PDF, reload the editor tab or restart LaTeX Workshop.

## Git collaboration workflow

1. Initialize the repo (if not already):
```bash
git init
git add .
git commit -m "Initial commit - LaTeX project"
```

2. Push to remote (GitHub example):
```bash
gh repo create <your-org-or-username>/<repo-name> --public --source=. --remote=origin
git push -u origin main
```

3. Branching workflow for collaborators:
- Create a feature branch: `git checkout -b feat/your-change`
- Make edits, build locally to verify (see build commands above).
- Commit logically: `git add` / `git commit -m "Short, descriptive message"`.
- Push and open a Pull Request to `main`.

4. Review and merge via PR, re-run build on CI (recommended).

## CI / Automated builds (optional)

You can add a CI workflow (GitHub Actions, GitLab CI) using a TeX Live image or MiKTeX to build the PDF and attach it to CI artifacts.

A simple GitHub Actions job (example): use `dante-ev/latex-action` or a TeX Live container and run the above build commands.

## Files of interest

- `main.tex` — main LaTeX source
- `references.bib` — BibTeX bibliography database
- `TP3 DL.tex` — other TeX file in the workspace
- `.vscode/settings.json` — project-specific LaTeX Workshop recipe (edit if your binaries are elsewhere)

## If something goes wrong

Paste the first 50 lines of the `xelatex` console output or `main.log` into an issue or the PR so collaborators can help. Common issues: missing fonts, missing perl for `latexmk`, undefined citations (run `bibtex`), or missing packages.

---

If you want, I can now initialize the git repository here and make an initial commit including `.gitignore`, `.vscode/settings.json`, and the README.
