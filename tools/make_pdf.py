"""Render ViscaBarcaVIT.md to a plain black-and-white PDF (needs `pip install markdown` and Edge or Chrome)."""
import os
import subprocess
import sys

import markdown

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSS = """
@page { size: A4; margin: 20mm 18mm; }
body { font-family: 'Times New Roman', serif; font-size: 11pt; line-height: 1.38; color: #000; background: #fff; }
h1 { font-size: 18pt; margin: 0 0 6pt; } h2 { font-size: 13.5pt; margin: 16pt 0 4pt; border-bottom: 1px solid #000; padding-bottom: 2pt; }
h3 { font-size: 11.5pt; margin: 10pt 0 2pt; } p, li { margin: 3pt 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0; font-size: 10pt; }
th, td { border: 1px solid #000; padding: 3pt 5pt; text-align: left; vertical-align: top; } th { font-weight: bold; }
code { font-family: 'Courier New', monospace; font-size: 9.5pt; }
pre { border: 1px solid #000; padding: 6pt; font-size: 9pt; white-space: pre-wrap; page-break-inside: avoid; }
a { color: #000; text-decoration: none; }
"""


def main():
    src = os.path.join(ROOT, "ViscaBarcaVIT.md")
    html = markdown.markdown(open(src, encoding="utf-8").read(), extensions=["tables", "fenced_code"])
    page = "<!doctype html><meta charset='utf-8'><title>ViscaBarcaVIT</title><style>%s</style>%s" % (CSS, html)
    tmp = os.path.join(ROOT, "cache", "doc.html")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    open(tmp, "w", encoding="utf-8").write(page)
    out = os.path.join(ROOT, "ViscaBarcaVIT.pdf")
    for exe in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                r"C:\Program Files\Google\Chrome\Application\chrome.exe"):
        if os.path.exists(exe):
            subprocess.run([exe, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                            "--print-to-pdf=" + out, "file:///" + tmp.replace("\\", "/")], check=True)
            print("wrote", out)
            return
    sys.exit("Edge or Chrome not found")


if __name__ == "__main__":
    main()
