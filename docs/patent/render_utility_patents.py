#!/usr/bin/env python3
"""Render the two utility-model patent drafts into portable HTML files.

The renderer intentionally supports the small Markdown subset used by the
drafts: headings, paragraphs, tables, block notes, numbered lists and figures.
Images are kept as relative files so LibreOffice can convert the HTML to PDF
without external network access.
"""

from __future__ import annotations

import html
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"

STYLE = r"""
@page { size: A4 portrait; margin: 22mm 21mm 22mm 24mm; }
html { background: #e8e8e8; }
body {
  width: 168mm; margin: 0 auto; background: #fff; color: #111;
  font-family: "Noto Serif CJK SC", "Source Han Serif SC", SimSun, serif;
  font-size: 10.5pt; line-height: 1.68; text-align: justify;
  padding: 18mm 21mm 24mm 24mm; box-shadow: 0 0 8px rgba(0,0,0,.15);
}
@media print { html { background: #fff; } body { width: auto; margin: 0; padding: 0; box-shadow: none; } }
h1 { font-family: "Noto Sans CJK SC", sans-serif; text-align: center; font-size: 18pt; margin: 1.2em 0 .9em; page-break-after: avoid; }
h2 { font-family: "Noto Sans CJK SC", sans-serif; text-align: center; font-size: 14pt; margin: 1.35em 0 .65em; page-break-after: avoid; }
h3 { font-family: "Noto Sans CJK SC", sans-serif; font-size: 11.5pt; margin: 1.2em 0 .55em; page-break-after: avoid; }
p { margin: .32em 0; orphans: 2; widows: 2; }
.claim { margin: .55em 0; }
.claim strong { font-family: "Noto Sans CJK SC", sans-serif; }
.draft-note { border: 1px solid #777; background: #fafafa; padding: 10px 13px; font-size: 9.5pt; }
table { border-collapse: collapse; width: 100%; margin: .75em 0 1em; font-size: 9.5pt; page-break-inside: avoid; }
th, td { border: 1px solid #555; padding: 5px 7px; vertical-align: top; }
th { background: #f1f1f1; font-family: "Noto Sans CJK SC", sans-serif; text-align: center; }
hr { border: none; border-top: 1px solid #666; margin: 1.4em 0; }
.figure { text-align: center; page-break-before: always; page-break-inside: avoid; margin: 0; padding-top: 2mm; }
.figure img { display: block; width: 100%; max-width: 100%; height: auto; max-height: 235mm; object-fit: contain; }
.caption { font-family: "Noto Sans CJK SC", sans-serif; font-size: 10pt; margin-top: 5mm; text-align: center; }
.pagebreak { page-break-before: always; }
ol { margin: .4em 0 .8em 2em; padding: 0; }
li { margin: .35em 0; }
code { font-family: "Noto Sans Mono CJK SC", monospace; font-size: .9em; background: #f4f4f4; padding: 0 2px; }
"""


def inline(value: str) -> str:
    escaped = html.escape(value, quote=False)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    return escaped


def table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def table_separator(line: str) -> bool:
    cells = table_cells(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def render(markdown: str) -> str:
    lines = markdown.splitlines()
    parts: list[str] = []
    paragraph: list[str] = []
    in_list = False

    def flush() -> None:
        nonlocal paragraph
        if not paragraph:
            return
        text = " ".join(line.strip() for line in paragraph)
        cls = ' class="claim"' if text.startswith("**") and re.match(r"\*\*\d+\.", text) else ""
        parts.append(f"<p{cls}>{inline(text)}</p>")
        paragraph = []

    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if not stripped:
            flush()
            if in_list:
                parts.append("</ol>")
                in_list = False
            i += 1
            continue
        if stripped == "---":
            flush()
            parts.append("<hr>")
            i += 1
            continue
        if stripped.startswith(">"):
            flush()
            parts.append(f'<div class="draft-note">{inline(stripped.lstrip("> "))}</div>')
            i += 1
            continue
        image = re.fullmatch(r"!\[([^]]+)\]\(([^)]+)\)", stripped)
        if image:
            flush()
            alt, src = image.groups()
            src_path = Path(src)
            if not src_path.is_absolute():
                src = "../" + src
            parts.append(
                f'<div class="figure"><img src="{html.escape(src, quote=True)}" alt="{html.escape(alt, quote=True)}">'
                f'<div class="caption">{inline(alt)}</div></div>'
            )
            i += 1
            continue
        heading = re.fullmatch(r"(#{1,3})\s+(.+)", stripped)
        if heading:
            flush()
            if in_list:
                parts.append("</ol>")
                in_list = False
            level, title = len(heading.group(1)), heading.group(2)
            klass = ' class="pagebreak"' if level == 1 and title in {"权利要求书", "说明书", "附图"} else ""
            parts.append(f"<h{level}{klass}>{inline(title)}</h{level}>")
            i += 1
            continue
        if stripped.startswith("|") and i + 1 < len(lines) and table_separator(lines[i + 1]):
            flush()
            if in_list:
                parts.append("</ol>")
                in_list = False
            headers = table_cells(stripped)
            parts.append("<table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in headers) + "</tr></thead><tbody>")
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = table_cells(lines[i])
                parts.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
                i += 1
            parts.append("</tbody></table>")
            continue
        list_match = re.match(r"^\d+\.\s+(.+)$", stripped)
        if list_match:
            flush()
            if not in_list:
                parts.append("<ol>")
                in_list = True
            parts.append(f"<li>{inline(list_match.group(1))}</li>")
            i += 1
            continue
        if in_list:
            parts.append("</ol>")
            in_list = False
        paragraph.append(raw)
        i += 1
    flush()
    if in_list:
        parts.append("</ol>")
    return "\n".join(parts)


def build(source: Path) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"{source.stem}.html"
    body = render(source.read_text(encoding="utf-8"))
    title = source.stem
    document = f"<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><title>{html.escape(title)}</title><style>{STYLE}</style></head><body>{body}</body></html>"
    target.write_text(document, encoding="utf-8")
    return target


def main() -> None:
    sources = [ROOT / "实用新型一_羽毛球动作证据反馈装置.md", ROOT / "实用新型二_骨骼与球路联合时序识别装置.md"]
    for source in sources:
        print(build(source))


if __name__ == "__main__":
    main()
