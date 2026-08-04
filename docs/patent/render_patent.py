#!/usr/bin/env python3
"""Render the patent draft Markdown into a self-contained styled HTML file.

The renderer intentionally supports only the small Markdown subset used by
the local draft.  It has no third-party dependencies, which keeps document
generation reproducible in the project environment.
"""

from __future__ import annotations

import html
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "羽毛球运动分析发明专利初稿.md"
TARGET = ROOT / "output" / "羽毛球运动分析发明专利初稿.html"


STYLE = r"""
@page { size: A4 portrait; margin: 22mm 21mm 22mm 24mm; }
html { background: #e7e7e7; }
body {
  width: 168mm; margin: 0 auto; background: white; color: #111;
  font-family: "Noto Serif CJK SC", "Source Han Serif SC", SimSun, serif;
  font-size: 10.5pt; line-height: 1.72; text-align: justify;
  padding: 18mm 21mm 24mm 24mm; box-shadow: 0 0 8px rgba(0,0,0,.15);
}
@media print {
  html { background: white; }
  body { width: auto; margin: 0; padding: 0; box-shadow: none; }
  a { color: black; text-decoration: none; }
}
h1 { font-family: "Noto Sans CJK SC", sans-serif; text-align: center; font-size: 18pt; margin: 1.2em 0 .9em; page-break-after: avoid; }
h2 { font-family: "Noto Sans CJK SC", sans-serif; text-align: center; font-size: 14pt; margin: 1.35em 0 .65em; page-break-after: avoid; }
h3 { font-family: "Noto Sans CJK SC", sans-serif; font-size: 11.5pt; margin: 1.2em 0 .55em; page-break-after: avoid; }
p { margin: .32em 0; orphans: 2; widows: 2; }
.numbered { text-indent: 0; }
.claim { margin: .5em 0; }
.claim strong { font-family: "Noto Sans CJK SC", sans-serif; }
.draft-note { border: 1px solid #777; background: #fafafa; padding: 10px 13px; font-size: 9.5pt; }
table { border-collapse: collapse; width: 100%; margin: .75em 0 1em; font-size: 9.5pt; page-break-inside: avoid; }
th, td { border: 1px solid #555; padding: 5px 7px; vertical-align: top; }
th { background: #f1f1f1; font-family: "Noto Sans CJK SC", sans-serif; text-align: center; }
hr { border: none; border-top: 1px solid #666; margin: 1.4em 0; }
.formula { text-align: center; font-family: "Noto Serif CJK SC", serif; white-space: pre-wrap; margin: .8em 0; page-break-inside: avoid; }
.figure { text-align: center; page-break-before: always; page-break-inside: avoid; margin: 0; padding-top: 2mm; }
.figure img { max-width: 100%; max-height: 235mm; object-fit: contain; }
.caption { font-family: "Noto Sans CJK SC", sans-serif; font-size: 10pt; margin-top: 5mm; text-align: center; }
.pagebreak { page-break-before: always; }
.cover { min-height: 225mm; display: flex; flex-direction: column; justify-content: center; }
.cover h1 { font-size: 21pt; line-height: 1.5; }
.cover .sub { text-align: center; font-size: 13pt; margin-top: 2em; }
ol { margin: .4em 0 .8em 2em; padding: 0; }
li { margin: .35em 0; }
code { font-family: "Noto Sans Mono CJK SC", monospace; font-size: .9em; background: #f4f4f4; padding: 0 2px; }
.legal { color: #333; }
"""


def inline(value: str) -> str:
    value = html.escape(value, quote=False)
    value = re.sub(r"`([^`]+)`", r"<code>\1</code>", value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", value)
    return value


def is_table_separator(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def render(markdown: str) -> str:
    lines = markdown.splitlines()
    parts: list[str] = []
    paragraph: list[str] = []
    in_formula = False
    formula: list[str] = []
    in_list = False
    figure_index = 0

    def flush_paragraph() -> None:
        nonlocal paragraph
        if not paragraph:
            return
        text = " ".join(item.strip() for item in paragraph)
        classes = []
        if re.match(r"^\[\d{4}\]", text):
            classes.append("numbered")
        if re.match(r"^\*\*\d+\.", text):
            classes.append("claim")
        cls = f' class="{" ".join(classes)}"' if classes else ""
        parts.append(f"<p{cls}>{inline(text)}</p>")
        paragraph = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped == "$$":
            flush_paragraph()
            if in_formula:
                parts.append('<div class="formula">' + html.escape("\n".join(formula)) + "</div>")
                formula = []
                in_formula = False
            else:
                in_formula = True
            i += 1
            continue
        if in_formula:
            formula.append(line)
            i += 1
            continue

        image_match = re.fullmatch(r"!\[([^]]+)\]\(([^)]+)\)", stripped)
        if image_match:
            flush_paragraph()
            if in_list:
                parts.append("</ol>")
                in_list = False
            figure_index += 1
            alt, source = image_match.groups()
            # The generated HTML lives in ``output/`` one level below the
            # Markdown source, while image paths in Markdown are source-local.
            if not re.match(r"^(?:[a-z]+:|/)", source, flags=re.IGNORECASE):
                source = "../" + source
            parts.append(
                f'<div class="figure"><img src="{html.escape(source, quote=True)}" alt="{html.escape(alt, quote=True)}">'
                f'<div class="caption">{inline(alt)}</div></div>'
            )
            i += 1
            continue

        if stripped.startswith("|") and i + 1 < len(lines) and is_table_separator(lines[i + 1]):
            flush_paragraph()
            if in_list:
                parts.append("</ol>")
                in_list = False
            headers = table_row(stripped)
            parts.append("<table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in headers) + "</tr></thead><tbody>")
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = table_row(lines[i])
                parts.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
                i += 1
            parts.append("</tbody></table>")
            continue

        list_match = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if list_match:
            flush_paragraph()
            if not in_list:
                parts.append("<ol>")
                in_list = True
            parts.append(f"<li>{inline(list_match.group(2))}</li>")
            i += 1
            continue
        elif in_list:
            parts.append("</ol>")
            in_list = False

        heading = re.match(r"^(#{1,3})\s+(.*)$", stripped)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            title = heading.group(2)
            extra = ""
            if level == 1 and title in {"权利要求书", "说明书", "说明书附图", "撰写校核附录（不纳入正式申请文件）"}:
                extra = ' class="pagebreak"'
            parts.append(f"<h{level}{extra}>{inline(title)}</h{level}>")
            i += 1
            continue

        if stripped.startswith(">"):
            flush_paragraph()
            note = stripped.lstrip("> ")
            parts.append(f'<div class="draft-note">{inline(note)}</div>')
            i += 1
            continue

        if stripped == "---":
            flush_paragraph()
            parts.append("<hr>")
            i += 1
            continue

        if not stripped:
            flush_paragraph()
        else:
            paragraph.append(line)
        i += 1

    flush_paragraph()
    if in_list:
        parts.append("</ol>")
    return "\n".join(parts)


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    content = render(SOURCE.read_text(encoding="utf-8"))
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>羽毛球运动分析国家发明专利初稿</title>
<style>{STYLE}</style>
</head>
<body>
{content}
</body>
</html>
"""
    TARGET.write_text(document, encoding="utf-8")
    print(TARGET)


if __name__ == "__main__":
    main()
