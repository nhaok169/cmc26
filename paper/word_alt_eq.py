# -*- coding: utf-8 -*-
"""把已有 Word 里残留的 $...$ / $$...$$ 批量转成公式，相当于反复 Alt+=。

用法（先关掉目标文档，避免占用）：
    python paper/word_alt_eq.py
    python paper/word_alt_eq.py 某文件.docx
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

MATH_RE = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$", re.S)

WD_DISPLAY = 0
WD_INLINE = 1


def convert_paragraph(doc, para) -> int:
    rng = para.Range
    raw = rng.Text or ""
    if raw.endswith("\r"):
        body = raw[:-1]
    else:
        body = raw
    matches = list(MATH_RE.finditer(body))
    n = 0
    base = rng.Start
    for m in reversed(matches):
        inner = (m.group(1) if m.group(1) is not None else m.group(2) or "").strip()
        if not inner:
            continue
        display = m.group(1) is not None
        fr = doc.Range(base + m.start(), base + m.end())
        fr.Text = inner
        try:
            om = fr.OMaths.Add(fr)
            om.BuildUp()
            om.Type = WD_DISPLAY if display else WD_INLINE
            n += 1
        except Exception as e:
            print("skip:", inner[:60].replace("\n", " "), e)
    return n


def convert_file(path: Path) -> int:
    import win32com.client

    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    doc = word.Documents.Open(str(path.resolve()))
    total = 0
    try:
        for i in range(doc.Paragraphs.Count, 0, -1):
            total += convert_paragraph(doc, doc.Paragraphs(i))
        doc.Save()
    finally:
        doc.Close(SaveChanges=True)
        # 不 Quit，以免关掉用户正在用的 Word
    return total


def main():
    here = Path(__file__).resolve().parent
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = here / "无线电干扰源的快速自动定位与清除.docx"
    if not path.is_file():
        raise SystemExit(f"找不到文件：{path}")
    print("converting", path)
    n = convert_file(path)
    print("converted", n, "equations")


if __name__ == "__main__":
    main()
