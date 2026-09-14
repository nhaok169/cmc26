# -*- coding: utf-8 -*-
"""把 paper/B题_问题1-3.md 转为国赛体例 Word。"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent / "B题_全文.md"
OUT = Path(__file__).resolve().parent / "无线电干扰源的快速自动定位与清除.docx"
EQ_DIR = Path(__file__).resolve().parent / "_eq"
EQ_DIR.mkdir(exist_ok=True)

CN_BODY = "宋体"
CN_HEAD = "黑体"
EN_FONT = "Times New Roman"
MATH_FONT = "Cambria Math"


def set_run_font(run, east=CN_BODY, ascii_font=EN_FONT, size=12, bold=False, italic=False, color=None):
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(size)
    run.font.name = ascii_font
    if color is not None:
        run.font.color.rgb = color
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.get_or_add_rFonts()
    rFonts.set(qn("w:ascii"), ascii_font)
    rFonts.set(qn("w:hAnsi"), ascii_font)
    rFonts.set(qn("w:eastAsia"), east)
    rFonts.set(qn("w:cs"), ascii_font)


def set_paragraph_format(p, *, first_line=None, before=0, after=0, line=1.5, align=None, east=CN_BODY, exact_pt=None):
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    if exact_pt is not None:
        pf.line_spacing = Pt(exact_pt)
        pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY
    else:
        pf.line_spacing = line
        pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    if first_line is None:
        pf.first_line_indent = Cm(0)
    else:
        pf.first_line_indent = Cm(first_line)
    if align is not None:
        p.alignment = align
    p.style.font.name = east
    pPr = p._p.get_or_add_pPr()
    rPr = pPr.find(qn("w:rPr"))
    if rPr is None:
        rPr = OxmlElement("w:rPr")
        pPr.append(rPr)
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:eastAsia"), east)


def _border_el(edge: str, val: str, sz: int, color: str = "000000"):
    el = OxmlElement(f"w:{edge}")
    el.set(qn("w:val"), val)
    el.set(qn("w:sz"), str(sz if val != "nil" else 0))
    el.set(qn("w:space"), "0")
    el.set(qn("w:color"), color)
    return el


def _clear_tbl_borders(table):
    tblPr = table._tbl.tblPr
    old = tblPr.find(qn("w:tblBorders"))
    if old is not None:
        tblPr.remove(old)
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        borders.append(_border_el(edge, "nil", 0))
    tblPr.append(borders)


def set_three_line_cell(cell, *, top: int | None = None, bottom: int | None = None):
    """科技论文三线表：仅顶线、栏线、底线，无竖线、无底纹。sz 单位为 1/8 磅。"""
    tcPr = cell._tc.get_or_add_tcPr()
    old = tcPr.find(qn("w:tcBorders"))
    if old is not None:
        tcPr.remove(old)
    shd = tcPr.find(qn("w:shd"))
    if shd is not None:
        tcPr.remove(shd)
    tcBorders = OxmlElement("w:tcBorders")
    tcBorders.append(_border_el("top", "single" if top else "nil", top or 0))
    tcBorders.append(_border_el("left", "nil", 0))
    tcBorders.append(_border_el("bottom", "single" if bottom else "nil", bottom or 0))
    tcBorders.append(_border_el("right", "nil", 0))
    tcPr.append(tcBorders)


def add_page_break(doc):
    p = doc.add_paragraph()
    set_paragraph_format(p, first_line=0, before=0, after=0, line=1.0)
    p.add_run().add_break(WD_BREAK.PAGE)


def set_row_cant_split(row):
    trPr = row._tr.get_or_add_trPr()
    if trPr.find(qn("w:cantSplit")) is None:
        trPr.append(OxmlElement("w:cantSplit"))


def set_row_header(row):
    trPr = row._tr.get_or_add_trPr()
    if trPr.find(qn("w:tblHeader")) is None:
        trPr.append(OxmlElement("w:tblHeader"))


def set_cell_nowrap(cell):
    tcPr = cell._tc.get_or_add_tcPr()
    if tcPr.find(qn("w:noWrap")) is None:
        tcPr.append(OxmlElement("w:noWrap"))


def add_page_number(paragraph):
    run1 = paragraph.add_run()
    fld1 = OxmlElement("w:fldChar")
    fld1.set(qn("w:fldCharType"), "begin")
    run1._r.append(fld1)
    run2 = paragraph.add_run()
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    run2._r.append(instr)
    run3 = paragraph.add_run()
    fld2 = OxmlElement("w:fldChar")
    fld2.set(qn("w:fldCharType"), "end")
    run3._r.append(fld2)
    for r in (run1, run2, run3):
        set_run_font(r, size=10.5)


GREEK = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "varepsilon": "ε",
    "theta": "θ",
    "kappa": "κ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "pi": "π",
    "rho": "ρ",
    "sigma": "σ",
    "phi": "φ",
    "varphi": "φ",
    "omega": "ω",
    "Gamma": "Γ",
    "Delta": "Δ",
    "Theta": "Θ",
    "Lambda": "Λ",
    "Sigma": "Σ",
    "Phi": "Φ",
    "Omega": "Ω",
}

SUB = str.maketrans("0123456789+-=()nijkxyabcdeilmprstuv", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₙᵢⱼₖₓᵧₐbcdₑᵢₗₘₚᵣₛₜᵤᵥ")
SUP = str.maketrans("0123456789+-=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ")


def _brace_arg(s: str, i: int) -> tuple[str, int]:
    assert s[i] == "{"
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1 : j], j + 1
    return s[i + 1 :], len(s)


def latex_to_unicode(tex: str) -> str:
    s = tex.strip()
    s = s.replace(r"^{\circ}", "°").replace(r"^\circ", "°")
    s = s.replace(r"\,", " ").replace(r"\;", " ").replace(r"\!", "")
    s = s.replace(r"\ ", " ")
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\bigl", "").replace(r"\bigr", "")
    s = s.replace(r"\lvert", "|").replace(r"\rvert", "|")
    s = s.replace(r"\|", "‖").replace(r"\Vert", "‖").replace(r"\vert", "|")
    s = s.replace(r"\cdot", "·").replace(r"\times", "×")
    s = s.replace(r"\pm", "±").replace(r"\mp", "∓")
    s = s.replace(r"\leq", "≤").replace(r"\le", "≤")
    s = s.replace(r"\geq", "≥").replace(r"\ge", "≥")
    s = s.replace(r"\leqslant", "≤").replace(r"\geqslant", "≥")
    s = s.replace(r"\neq", "≠").replace(r"\equiv", "≡")
    s = s.replace(r"\approx", "≈").replace(r"\sim", "∼")
    s = s.replace(r"\propto", "∝").replace(r"\infty", "∞")
    s = s.replace(r"\in", "∈").replace(r"\notin", "∉")
    s = s.replace(r"\cap", "∩").replace(r"\cup", "∪")
    s = s.replace(r"\subset", "⊂").replace(r"\subseteq", "⊆")
    s = s.replace(r"\emptyset", "∅").replace(r"\varnothing", "∅")
    s = s.replace(r"\to", "→").replace(r"\rightarrow", "→")
    s = s.replace(r"\mapsto", "↦").replace(r"\Rightarrow", "⇒")
    s = s.replace(r"\perp", "⊥").replace(r"\angle", "∠")
    s = s.replace(r"\triangle", "△").replace(r"\circ", "°")
    s = s.replace(r"\partial", "∂").replace(r"\nabla", "∇")
    s = s.replace(r"\sum", "∑").replace(r"\prod", "∏")
    s = s.replace(r"\int", "∫").replace(r"\max", "max").replace(r"\min", "min")
    s = s.replace(r"\sin", "sin").replace(r"\cos", "cos").replace(r"\tan", "tan")
    s = s.replace(r"\arg", "arg").replace(r"\det", "det")
    s = s.replace(r"\inf", "inf").replace(r"\sup", "sup")
    s = s.replace(r"\bigcap", "⋂")
    s = s.replace(r"\gtrsim", "≳").replace(r"\lesssim", "≲")
    s = s.replace(r"\quad", "  ").replace(r"\qquad", "    ")
    s = s.replace(r"\mathrm{MEC}", "MEC").replace(r"\mathrm{m}", "m")
    s = s.replace(r"\mathrm{OK}", "OK")
    for name, ch in sorted(GREEK.items(), key=lambda kv: -len(kv[0])):
        s = s.replace("\\" + name, ch)
    s = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathcal\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\hat\{([^}]*)\}", r"\1̂", s)
    s = re.sub(r"\\hat([A-Za-z])", r"\1̂", s)
    s = re.sub(r"\\vec\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", s)

    def repl_frac(m):
        a, b = m.group(1), m.group(2)
        return f"({a})/({b})" if len(a) > 1 or len(b) > 1 else f"{a}/{b}"

    for _ in range(8):
        ns = re.sub(r"\\dfrac\{([^{}]*)\}\{([^{}]*)\}", repl_frac, s)
        ns = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", repl_frac, ns)
        ns = re.sub(r"\\sqrt\{([^{}]*)\}", r"√(\1)", ns)
        if ns == s:
            break
        s = ns
    # sub / sup of braced or single token
    def sub_repl(m):
        body = m.group(1)
        t = body.translate(SUB)
        return t if t != body or all(c in "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎" or not c.isalnum() for c in t) else "_" + body

    s = re.sub(r"_\{([^{}]+)\}", lambda m: m.group(1).translate(SUB), s)
    s = re.sub(r"\^\{([^{}]+)\}", lambda m: m.group(1).translate(SUP), s)
    s = re.sub(r"_([A-Za-z0-9])", lambda m: m.group(1).translate(SUB), s)
    s = re.sub(r"\^([A-Za-z0-9+\-])", lambda m: m.group(1).translate(SUP), s)
    s = s.replace("{", "").replace("}", "")
    s = s.replace("\\", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def mathtext_prep(tex: str) -> str:
    s = tex.strip()
    s = s.replace(r"\lvert", r"|").replace(r"\rvert", r"|")
    s = s.replace(r"\|", r"\Vert ")
    s = s.replace(r"\bigl", "").replace(r"\bigr", "")
    s = s.replace(r"\gtrsim", r"\geq ").replace(r"\lesssim", r"\leq ")
    s = s.replace(r"\dfrac", r"\frac")
    s = s.replace(r"\mathbf", r"\mathrm")
    s = s.replace(r"\mathcal", r"\mathrm")
    s = s.replace(r"\,", r"\;")
    s = re.sub(r"\\le(?!qslant|q)", r"\\leq ", s)
    s = re.sub(r"\\ge(?!qslant|q)", r"\\geq ", s)
    return s


def render_display_eq(tex: str) -> Path | None:
    key = hashlib.md5(tex.encode("utf-8")).hexdigest()[:12]
    path = EQ_DIR / f"eq_{key}.png"
    if path.exists() and path.stat().st_size > 200:
        return path
    prepared = mathtext_prep(tex)
    try:
        fig = plt.figure(figsize=(6.8, 0.72))
        fig.patch.set_facecolor("white")
        fig.text(0.5, 0.52, f"${prepared}$", ha="center", va="center", fontsize=13, color="#1F1F1F")
        plt.axis("off")
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white", pad_inches=0.08)
        plt.close(fig)
        if path.exists() and path.stat().st_size > 200:
            return path
    except Exception as e:
        plt.close("all")
        print("eq render fallback:", e, "tex=", prepared[:80])
    return None


TOKEN_RE = re.compile(r"(\$\$.*?\$\$|\$.*?\$|\*\*[^*]+\*\*)", re.S)


def append_omath(p, tex: str, *, display: bool = False, size=12):
    """把 LaTeX 写成 Word 原生公式（OMML），无需再选中 Alt+=。"""
    from latex_omml import omml_element

    try:
        p._p.append(omml_element(tex, display=display))
        return True
    except Exception as e:
        print("math fallback:", e, "tex=", tex[:80].replace("\n", " "))
        body = f"$${tex}$$" if display else f"${tex}$"
        run = p.add_run(body)
        set_run_font(run, east=EN_FONT, ascii_font="Cambria Math", size=size, italic=False)
        return False


def add_mixed_runs(p, text: str, *, size=12, east=CN_BODY, bold=False, first_math=False):
    """正文汉字用宋体；`$...$` / `$$...$$` 转为 Word 公式对象。"""
    if not text:
        return
    parts = re.split(r"(\$\$.*?\$\$|\$(?:\\\$|[^$])+\$|\*\*[^*]+\*\*)", text)
    for part in parts:
        if not part:
            continue
        if part.startswith("$$") and part.endswith("$$") and len(part) >= 4:
            append_omath(p, part[2:-2], display=True, size=size)
        elif part.startswith("$") and part.endswith("$") and len(part) >= 2:
            append_omath(p, part[1:-1], display=False, size=size)
        elif part.startswith("**") and part.endswith("**"):
            run = p.add_run(part[2:-2])
            set_run_font(run, east=east, size=size, bold=True)
        else:
            run = p.add_run(part)
            set_run_font(run, east=east, size=size, bold=bold)


def add_body(doc, text: str, *, first=True, before=0, after=2, size=12, exact_pt=None):
    p = doc.add_paragraph()
    set_paragraph_format(
        p,
        first_line=0.74 if first else 0,
        before=before,
        after=after,
        line=1.25,
        exact_pt=16 if exact_pt is None else exact_pt,
    )
    add_mixed_runs(p, text, size=size)
    return p


def add_caption(doc, text: str):
    p = doc.add_paragraph()
    set_paragraph_format(p, first_line=0, before=1, after=3, line=1.1, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_mixed_runs(p, text, size=10)
    return p


def add_heading_cn(doc, text: str, level: int):
    p = doc.add_paragraph()
    if level == 1:
        set_paragraph_format(p, first_line=0, before=6, after=3, line=1.25, east=CN_HEAD)
        add_mixed_runs(p, text, size=14, east=CN_HEAD, bold=True)
    elif level == 2:
        set_paragraph_format(p, first_line=0, before=6, after=3, line=1.25, east=CN_HEAD)
        add_mixed_runs(p, text, size=12, east=CN_HEAD, bold=True)
    else:
        set_paragraph_format(p, first_line=0, before=5, after=2, line=1.25, east=CN_HEAD)
        add_mixed_runs(p, text, size=12, east=CN_HEAD, bold=True)
    return p


def add_display_math(doc, tex: str):
    """独立公式：写入居中的 Word 公式对象。"""
    p = doc.add_paragraph()
    set_paragraph_format(p, first_line=0, before=3, after=3, line=1.1, align=WD_ALIGN_PARAGRAPH.CENTER)
    body = tex.strip()
    if body.startswith("$$") and body.endswith("$$"):
        body = body[2:-2].strip()
    append_omath(p, body, display=True, size=11)
    return p


def add_image(doc, path: Path, width_cm=10.0):
    if not path.exists():
        add_body(doc, f"（缺图：{path}）", first=False)
        return
    compact = {
        "fig_q1_shape_ab.png": (8.0, 3),
        "fig_q1_shape_cd.png": (8.0, 3),
        "fig_q4_cert26.png": (9.0, 4),
    }
    before = 5
    if path.name in compact:
        width_cm, before = compact[path.name]
    p = doc.add_paragraph()
    set_paragraph_format(p, first_line=0, before=before, after=2, line=1.0, align=WD_ALIGN_PARAGRAPH.CENTER)
    run = p.add_run()
    run.add_picture(str(path), width=Cm(width_cm))


def _set_table_widths(table, widths_cm: list[float]):
    table.autofit = False
    tbl = table._tbl
    tblPr = tbl.tblPr
    if tblPr is None:
        tblPr = OxmlElement("w:tblPr")
        tbl.insert(0, tblPr)
    tblW = tblPr.find(qn("w:tblW"))
    if tblW is None:
        tblW = OxmlElement("w:tblW")
        tblPr.append(tblW)
    total = int(sum(widths_cm) * 567)
    tblW.set(qn("w:w"), str(total))
    tblW.set(qn("w:type"), "dxa")
    grid = tbl.find(qn("w:tblGrid"))
    if grid is not None:
        for i, child in enumerate(list(grid)):
            if i < len(widths_cm):
                child.set(qn("w:w"), str(int(widths_cm[i] * 567)))
    for row in table.rows:
        for i, w in enumerate(widths_cm):
            if i < len(row.cells):
                row.cells[i].width = Cm(w)


def add_table(doc, header: list[str], rows: list[list[str]], title: str | None = None):
    if title:
        p = doc.add_paragraph()
        set_paragraph_format(p, first_line=0, before=4, after=2, line=1.1, align=WD_ALIGN_PARAGRAPH.CENTER)
        add_mixed_runs(p, title, size=10.5, bold=True)
    table = doc.add_table(rows=1 + len(rows), cols=len(header))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    _clear_tbl_borders(table)
    is_symbol = any("首次出现" in (h or "") for h in header)
    n_data = len(rows)
    thick, thin = 12, 6  # 1.5 pt / 0.75 pt
    for j, h in enumerate(header):
        cell = table.rows[0].cells[j]
        cell.text = ""
        p = cell.paragraphs[0]
        set_paragraph_format(p, first_line=0, before=2, after=2, line=1.0, align=WD_ALIGN_PARAGRAPH.CENTER)
        add_mixed_runs(p, h, size=9, bold=True)
        set_three_line_cell(cell, top=thick, bottom=thin)
        if is_symbol and j == len(header) - 1:
            set_cell_nowrap(cell)
    set_row_header(table.rows[0])
    set_row_cant_split(table.rows[0])
    for i, row in enumerate(rows):
        ncols = len(header)
        while len(row) < ncols:
            row.append("")
        row = row[:ncols]
        set_row_cant_split(table.rows[i + 1])
        is_last = i == n_data - 1
        for j, val in enumerate(row):
            cell = table.rows[i + 1].cells[j]
            cell.text = ""
            p = cell.paragraphs[0]
            set_paragraph_format(p, first_line=0, before=1, after=1, line=1.0, align=WD_ALIGN_PARAGRAPH.CENTER)
            add_mixed_runs(p, val, size=8.5 if is_symbol else 9)
            set_three_line_cell(cell, top=None, bottom=thick if is_last else None)
            if is_symbol and j == ncols - 1:
                set_cell_nowrap(cell)
    if is_symbol and len(header) == 4:
        _set_table_widths(table, [3.2, 8.0, 1.4, 2.6])
    elif len(header) == 5:
        _set_table_widths(table, [4.0, 1.8, 2.6, 3.0, 2.8])
    p = doc.add_paragraph()
    set_paragraph_format(p, first_line=0, before=0, after=3, line=1.0)


def add_code_file(doc, rel: str):
    path = ROOT / rel.replace("\\", "/").strip()
    p = doc.add_paragraph()
    set_paragraph_format(p, first_line=0, before=8, after=4, line=1.25)
    run = p.add_run(f"程序 {rel}")
    set_run_font(run, east=CN_HEAD, size=10.5, bold=True)
    if not path.exists():
        add_body(doc, f"（缺文件：{rel}）", first=False)
        return
    raw = path.read_text(encoding="utf-8")
    if raw.startswith("\ufeff"):
        raw = raw[1:]
    raw = re.sub(r"202601006115", "YOUR_TEAM_ID", raw)
    lines = raw.replace("\t", "    ").splitlines()
    chunk: list[str] = []
    nchar = 0
    def flush(buf: list[str]):
        if not buf:
            return
        para = doc.add_paragraph()
        set_paragraph_format(para, first_line=0, before=0, after=0, line=1.0)
        run = para.add_run("\n".join(buf))
        set_run_font(run, east=CN_BODY, ascii_font="Consolas", size=8)
        pf = para.paragraph_format
        pf.left_indent = Cm(0.2)
    for ln in lines:
        extra = len(ln) + 1
        if nchar + extra > 3500 and chunk:
            flush(chunk)
            chunk, nchar = [], 0
        chunk.append(ln)
        nchar += extra
    flush(chunk)
    p = doc.add_paragraph()
    set_paragraph_format(p, first_line=0, before=0, after=6, line=1.0)


def split_table_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    cells: list[str] = []
    buf: list[str] = []
    in_math = False
    for c in s:
        if c == "$":
            in_math = not in_math
            buf.append(c)
        elif c == "|" and not in_math:
            cells.append("".join(buf).strip())
            buf = []
        else:
            buf.append(c)
    cells.append("".join(buf).strip())
    return cells


def parse_table(lines: list[str], i: int) -> tuple[list[str], list[list[str]], int]:
    rows = []
    while i < len(lines) and lines[i].startswith("|"):
        rows.append(split_table_row(lines[i]))
        i += 1
    if len(rows) >= 2 and all(re.fullmatch(r":?-{3,}:?", c.replace(" ", "")) for c in rows[1]):
        header, data = rows[0], rows[2:]
    else:
        header, data = rows[0], rows[1:]
    return header, data, i


def setup_doc() -> Document:
    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.left_margin = Cm(2.54)
    sec.right_margin = Cm(2.54)
    sec.top_margin = Cm(2.54)
    sec.bottom_margin = Cm(2.54)
    style = doc.styles["Normal"]
    style.font.name = EN_FONT
    style.font.size = Pt(12)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), CN_BODY)
    footer = sec.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_paragraph_format(p, first_line=0, before=0, after=0, line=1.0, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_page_number(p)
    return doc


def add_title_block(doc):
    p = doc.add_paragraph()
    set_paragraph_format(p, first_line=0, before=18, after=14, line=1.3, align=WD_ALIGN_PARAGRAPH.CENTER, east=CN_HEAD)
    run = p.add_run("无线电干扰源的快速自动定位与清除")
    set_run_font(run, east=CN_HEAD, size=18, bold=True)


def convert(src: Path = SRC, out: Path = OUT):
    text = src.read_text(encoding="utf-8")
    lines = text.splitlines()
    doc = setup_doc()
    add_title_block(doc)

    i = 0
    n = len(lines)
    pending_table_title = None
    skip_h1 = True  # markdown 首行标题已写入封面
    in_abstract = False

    while i < n:
        line = lines[i].rstrip()
        stripped = line.strip()

        if not stripped:
            i += 1
            continue
        if stripped == "---":
            i += 1
            continue

        if stripped.startswith("# "):
            if skip_h1:
                skip_h1 = False
                i += 1
                continue
            add_heading_cn(doc, stripped[2:].strip(), 1)
            i += 1
            continue
        if stripped.startswith("## "):
            title = stripped[3:].strip()
            if title == "摘要":
                in_abstract = True
                p = doc.add_paragraph()
                set_paragraph_format(p, first_line=0, before=8, after=6, line=1.5, align=WD_ALIGN_PARAGRAPH.CENTER, east=CN_HEAD)
                run = p.add_run("摘  要")
                set_run_font(run, east=CN_HEAD, size=14, bold=True)
            else:
                in_abstract = False
                if title == "附录":
                    add_page_break(doc)
                add_heading_cn(doc, title, 1)
            i += 1
            continue
        if stripped.startswith("### "):
            add_heading_cn(doc, stripped[4:].strip(), 2)
            i += 1
            continue
        if stripped.startswith("#### "):
            add_heading_cn(doc, stripped[5:].strip(), 3)
            i += 1
            continue

        if stripped.startswith("![") and "](" in stripped:
            m = re.search(r"!\[(.*?)\]\((.*?)\)", stripped)
            cap_alt = m.group(1) if m else ""
            rel = m.group(2) if m else ""
            img_path = (src.parent / rel).resolve()
            add_image(doc, img_path)
            i += 1
            # 紧随的加粗图题
            if i < n and lines[i].strip().startswith("**图"):
                add_caption(doc, lines[i].strip().replace("**", ""))
                i += 1
            elif cap_alt:
                add_caption(doc, cap_alt)
            continue

        if stripped.startswith("<!-- include-code:"):
            rel = stripped.replace("<!-- include-code:", "").replace("-->", "").strip()
            add_code_file(doc, rel)
            i += 1
            continue

        if stripped.startswith(">"):
            i += 1
            continue

        if stripped.startswith("|"):
            header, rows, i = parse_table(lines, i)
            title = pending_table_title
            pending_table_title = None
            add_table(doc, header, rows, title)
            continue

        if stripped.startswith("$$"):
            buf = [stripped]
            if not stripped.endswith("$$") or stripped == "$$":
                if stripped == "$$":
                    buf = []
                i += 1
                while i < n:
                    buf.append(lines[i])
                    if lines[i].strip().endswith("$$"):
                        break
                    i += 1
            tex = "\n".join(buf)
            tex = tex.strip()
            if tex.startswith("$$"):
                tex = tex[2:]
            if tex.endswith("$$"):
                tex = tex[:-2]
            tex = tex.strip()
            add_display_math(doc, tex)
            i += 1
            continue

        # 段落：合并后续非结构行
        para = [stripped]
        i += 1
        while i < n:
            nxt = lines[i].rstrip()
            ns = nxt.strip()
            if not ns:
                break
            if ns.startswith(("#", "|", "!", "$$", "---", "###", "####")):
                break
            if ns.startswith("**图") or ns.startswith("**表"):
                break
            para.append(ns)
            i += 1
        body = "".join(para) if False else "".join(para)
        # 中文段落：行末直接拼接；若原文以空格分隔的英文则保留单空格
        body = re.sub(r"\s+", " ", " ".join(para)).strip()
        # 中文之间的空格去掉（由换行引入）
        body = re.sub(r"(?<=[\u4e00-\u9fff]) (?=[\u4e00-\u9fff])", "", body)
        body = re.sub(r"(?<=[\u4e00-\u9fff]) (?=[，。；：、）】》])", "", body)
        body = re.sub(r"(?<=[（【《]) (?=[\u4e00-\u9fff])", "", body)

        if body.startswith("**表") and body.endswith("**"):
            pending_table_title = body.replace("**", "")
            continue
        if body.startswith("**关键词"):
            in_abstract = False
            p = doc.add_paragraph()
            set_paragraph_format(p, first_line=0.74, before=6, after=4, exact_pt=17)
            add_mixed_runs(p, body, size=12)
            add_page_break(doc)
            continue
        if body.startswith("**附录"):
            add_heading_cn(doc, body.replace("**", ""), 2)
            continue

        first = not body.startswith("**图")
        if re.match(r"^H\d+", body):
            add_body(doc, body, first=False, before=1, after=1, exact_pt=17 if in_abstract else None)
        else:
            add_body(doc, body, first=first, exact_pt=17 if in_abstract else None)
        continue

    out.parent.mkdir(exist_ok=True)
    try:
        doc.save(str(out))
        saved = out
    except PermissionError:
        saved = out.with_name(out.stem + "_公式.docx")
        doc.save(str(saved))
        print("原文件被 Word 占用，已另存为", saved.name)
    print("wrote", saved, "size", saved.stat().st_size)


if __name__ == "__main__":
    convert()
