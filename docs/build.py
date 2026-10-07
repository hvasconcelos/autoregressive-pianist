"""Build the EPUB of the book from the Markdown sources in this folder.

    uv run python docs/build.py                       # writes docs/build/The-Autoregressive-Pianist.epub
    uv run python docs/build.py --out book.epub       # choose the output file
    uv run python docs/build.py --format html         # one standalone HTML page, handy for proofreading

The Markdown files in docs/chapters are the source of truth. Each one becomes
one file inside the EPUB, in file-name order, so the numeric prefix sets the
reading order. Book metadata lives in docs/book.yaml and styling in
docs/epub.css. Needs pandoc 3 on the PATH (`brew install pandoc`).
"""
import argparse                                  # command-line options
import pathlib                                   # path handling
import shutil                                    # to find the pandoc binary
import subprocess                                # to run pandoc
import sys                                       # to exit with an error code

DOCS = pathlib.Path(__file__).resolve().parent   # the docs/ folder, wherever we are run from


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])        # CLI parser, described by the docstring's first line
    ap.add_argument("--out", type=pathlib.Path, default=None,                 # where to write the result
                    help="output file (default: docs/build/The-Autoregressive-Pianist.<ext>)")
    ap.add_argument("--format", choices=["epub", "html"], default="epub",     # EPUB for reading, HTML for quick proofreading
                    help="output format")
    args = ap.parse_args()                                                    # read the command line

    if shutil.which("pandoc") is None:                                        # pandoc does all the real work
        sys.exit("pandoc not found; install it with `brew install pandoc`")   # stop with a clear message

    chapters = sorted((DOCS / "chapters").glob("*.md"))                       # reading order = file-name order
    if not chapters:                                                          # nothing to build from
        sys.exit("no chapters found in docs/chapters")                        # stop with a clear message

    ext = "epub" if args.format == "epub" else "html"                         # file extension for the chosen format
    out = (args.out or DOCS / "build" / f"The-Autoregressive-Pianist.{ext}").resolve()  # absolute, since pandoc runs from docs/
    out.parent.mkdir(parents=True, exist_ok=True)                             # make sure the output folder exists

    cmd = [
        "pandoc",                                                             # the converter
        "--from", "markdown-smart",                                           # keep straight quotes exactly as typed
        "--metadata-file", "book.yaml",                                       # title, cover, language, identifier, css
        "--resource-path", ".:chapters",                                      # resolve ../images/... relative to the chapter files
        "--toc", "--toc-depth", "2",                                          # contents with chapters and their sections
        "--split-level", "1",                                                 # one EPUB file per top-level heading, i.e. per chapter
        "--highlight-style", "tango",                                         # colours for the code listings
        "--columns", "10000",                                                 # so long table rows do not force fixed column widths
        "--output", str(out),                                                 # where to write
    ]
    if args.format == "html":                                                 # a single self-contained page
        cmd += ["--standalone", "--embed-resources", "--css", "epub.css"]     # inline the images and stylesheet
    cmd += [str(p.relative_to(DOCS)) for p in chapters]                       # the chapter files, in order

    subprocess.run(cmd, cwd=DOCS, check=True)                                 # run pandoc from docs/ so book.yaml paths resolve
    print(f"wrote {out} from {len(chapters)} chapter files")                  # report what was built


if __name__ == "__main__":
    main()                                                                    # run when called as a script
