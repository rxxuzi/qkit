"""Report generator producing HTML and Markdown output.

Collects text sections and plotly figures, then writes them out as
self-contained HTML (with embedded charts) or Markdown files.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go


class ReportGenerator:
    """Build an analysis report incrementally and export it.

    Parameters
    ----------
    title : str
        Report title shown in the header.
    """

    def __init__(self, title: str = "Quant Analysis Report"):
        self.title = title
        self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        self._sections: list[dict] = []

    def add_section(self, heading: str, body: str):
        """Append a plain-text section."""
        self._sections.append({"type": "text", "heading": heading, "body": body})

    def add_figure(self, caption: str, fig: go.Figure):
        """Append a plotly figure."""
        self._sections.append({"type": "figure", "caption": caption, "fig": fig})

    def add_table(self, heading: str, markdown_table: str):
        """Append a Markdown-formatted table."""
        self._sections.append({"type": "table", "heading": heading, "body": markdown_table})

    # -- Markdown output ------------------------------------------------------

    def save_markdown(self, path: str):
        """Write the report as a Markdown file."""
        lines = [f"# {self.title}", f"*Generated: {self.created_at}*", ""]

        for sec in self._sections:
            if sec["type"] == "text":
                lines += [f"## {sec['heading']}", sec["body"], ""]
            elif sec["type"] == "table":
                lines += [f"## {sec['heading']}", sec["body"], ""]
            elif sec["type"] == "figure":
                fig_path = path.replace(".md", f"_{sec['caption'].replace(' ', '_')}.html")
                sec["fig"].write_html(fig_path)
                lines += [f"## {sec['caption']}",
                          f"[Interactive Chart]({Path(fig_path).name})", ""]

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("\n".join(lines), encoding="utf-8")

    # -- HTML output ----------------------------------------------------------

    def save_html(self, path: str):
        """Write the report as a self-contained HTML file with embedded charts."""
        parts = [
            "<!DOCTYPE html><html><head>",
            f'<meta charset="utf-8"><title>{self.title}</title>',
            "<style>",
            "body{font-family:'Segoe UI',Arial,sans-serif;max-width:1100px;"
            "margin:0 auto;padding:40px;background:#1a1a2e;color:#eee}",
            "h1{color:#00d2ff;border-bottom:2px solid #00d2ff;padding-bottom:10px}",
            "h2{color:#7fdbff;margin-top:40px}",
            "pre{background:#16213e;padding:16px;border-radius:8px;overflow-x:auto;font-size:13px}",
            "table{border-collapse:collapse;width:100%;margin:16px 0}",
            "th,td{border:1px solid #334;padding:8px 12px;text-align:left}",
            "th{background:#16213e;color:#7fdbff}",
            "tr:nth-child(even){background:#16213e40}",
            ".meta{color:#888;font-size:14px}",
            "</style>",
            '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>',
            "</head><body>",
            f"<h1>{self.title}</h1>",
            f'<p class="meta">Generated: {self.created_at}</p>',
        ]

        for i, sec in enumerate(self._sections):
            if sec["type"] == "text":
                parts += [f"<h2>{sec['heading']}</h2>",
                          f"<pre>{sec['body']}</pre>"]
            elif sec["type"] == "table":
                parts += [f"<h2>{sec['heading']}</h2>",
                          _md_table_to_html(sec["body"])]
            elif sec["type"] == "figure":
                div_id = f"chart_{i}"
                parts += [
                    f"<h2>{sec['caption']}</h2>",
                    f'<div id="{div_id}"></div>',
                    "<script>",
                    f"Plotly.newPlot('{div_id}',"
                    f"{sec['fig'].to_json()}.data,"
                    f"{sec['fig'].to_json()}.layout);",
                    "</script>",
                ]

        parts.append("</body></html>")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("\n".join(parts), encoding="utf-8")


def _md_table_to_html(md: str) -> str:
    """Minimal Markdown table -> HTML conversion."""
    lines = [l.strip() for l in md.strip().splitlines() if l.strip()]
    if len(lines) < 2:
        return f"<pre>{md}</pre>"

    html = ["<table>"]
    for i, line in enumerate(lines):
        if line.startswith("|--") or line.startswith("| --"):
            continue
        cells = [c.strip() for c in line.split("|") if c.strip()]
        tag = "th" if i == 0 else "td"
        html.append("  <tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
    html.append("</table>")
    return "\n".join(html)
