# -*- coding: utf-8 -*-
"""LaTeX → Word 公式（OMML）。用 Office 自带的 MML2OMML.XSL，生成后不必再 Alt+=。"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from lxml import etree

NS_M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

XSL_CANDIDATES = [
    Path(r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL"),
    Path(r"C:\Program Files (x86)\Microsoft Office\Office16\MML2OMML.XSL"),
    Path(r"C:\Program Files\Microsoft Office\Office16\MML2OMML.XSL"),
    Path(__file__).resolve().parent / "MML2OMML.XSL",
]


def find_xsl() -> Path:
    for p in XSL_CANDIDATES:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "找不到 MML2OMML.XSL。请确认已安装 Microsoft Word，"
        "或把该文件复制到 paper/ 目录。"
    )


@lru_cache(maxsize=1)
def _transform():
    return etree.XSLT(etree.parse(str(find_xsl())))


def prep_tex(tex: str) -> str:
    s = tex.strip()
    s = s.replace(r"^{\circ}", "^{°}").replace(r"^\circ", "^{°}")
    s = s.replace(r"\lvert", r"\left|").replace(r"\rvert", r"\right|")
    s = s.replace(r"\|", r"\Vert ")
    s = s.replace(r"\qquad", r"\quad\quad")
    s = s.replace(r"\dfrac", r"\frac")
    s = s.replace(r"\gtrsim", r"\geq")
    s = re.sub(r"\\hat\s+([A-Za-z])", r"\\hat{\1}", s)
    return s


@lru_cache(maxsize=512)
def latex_to_omml_xml(tex: str, *, display: bool = False) -> str:
    from latex2mathml.converter import convert

    mml = convert(prep_tex(tex), display="block" if display else "inline")
    root = etree.fromstring(mml.encode("utf-8"))
    omml = _transform()(root)
    node = omml.getroot()
    xml = etree.tostring(node, encoding="unicode")
    if "oMath" not in xml:
        raise RuntimeError("OMML 转换结果不含 oMath")
    return xml


def omml_element(tex: str, *, display: bool = False):
    """返回可插入 python-docx 段落的 OMML 元素。"""
    from docx.oxml import parse_xml

    xml = latex_to_omml_xml(tex, display=display)
    if display and xml.strip().startswith("<m:oMath ") and "oMathPara" not in xml:
        xml = (
            f'<m:oMathPara xmlns:m="{NS_M}">'
            f'<m:oMathParaPr><m:jc m:val="center"/></m:oMathParaPr>'
            f"{xml}"
            f"</m:oMathPara>"
        )
    return parse_xml(xml)
