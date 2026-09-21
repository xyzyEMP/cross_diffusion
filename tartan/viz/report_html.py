"""Render evaluation Markdown as a self-contained styled HTML report."""

from __future__ import annotations

from pathlib import Path

import markdown


STYLE = """
:root { color-scheme: light; }
body { margin: 0; background: #f4f6f8; color: #1f2937; font: 16px/1.65 Inter, system-ui, sans-serif; }
main { max-width: 1050px; margin: 36px auto; padding: 44px 52px; background: white; border-radius: 14px;
       box-shadow: 0 8px 30px rgba(15, 23, 42, .09); }
h1 { margin-top: 0; color: #14213d; font-size: 2rem; border-bottom: 3px solid #e5e7eb; padding-bottom: 14px; }
h2 { margin-top: 2rem; color: #1d4e89; font-size: 1.35rem; }
table { width: 100%; border-collapse: collapse; margin: 18px 0 26px; font-variant-numeric: tabular-nums; }
th { background: #1d4e89; color: white; text-align: left; }
th, td { padding: 10px 12px; border: 1px solid #d9e0e7; }
tbody tr:nth-child(even) { background: #f6f9fc; }
tbody tr:hover { background: #eaf2fb; }
code { background: #eef2f7; padding: 2px 6px; border-radius: 5px; }
img { display: block; max-width: 100%; height: auto; margin: 18px auto 26px; }
blockquote { margin: 18px 0; padding: 10px 18px; border-left: 4px solid #1d4e89; background: #f6f9fc; }
li { margin: 4px 0; }
@media (max-width: 760px) { main { margin: 0; padding: 24px 18px; border-radius: 0; } table { font-size: 13px; } }
"""


def render_report(markdown_path: Path, html_path: Path) -> None:
    source = markdown_path.read_text(encoding="utf-8")
    content = markdown.markdown(source, extensions=["tables", "fenced_code"])
    html_path.write_text(
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>TartanGround Evaluation Report</title><style>{STYLE}</style></head>"
        f"<body><main>{content}</main></body></html>",
        encoding="utf-8",
    )
