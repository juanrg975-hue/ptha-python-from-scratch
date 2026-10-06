"""Screenshots of every card and section of kermadectonga2_v9/report.html, for
report_guide.html, and of the blocks a segmented report adds, from
kermadectonga2_v9seg/report.html (saved as report_seg_<k>_<slug>.jpg). Only
READS the example folders.

Each block of the report (a STEP card or a section) is put alone on a page
with the report's own style, screenshotted with headless Edge, trimmed and
saved as html/docs/img/report_<k>_<slug>.jpg. The map card loads Esri tiles,
so this needs an internet connection.

Run from ptha18_logic_tree_test/:
    .venv/Scripts/python.exe from_scratch_v12/html/docs/figures_src/make_report_guide.py
"""
import html
import os
import re
import subprocess
import tempfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(DOCS, "..", "..", ".."))
EDGE = r"C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
REPORT = os.path.join(ROOT, "kermadectonga2_v9", "report.html")
REPORT_SEG = os.path.join(ROOT, "kermadectonga2_v9seg", "report.html")

doc = open(REPORT, encoding="utf-8").read()
style = re.search(r"<style>.*?</style>", doc, re.S).group(0)
body = doc[doc.index('<div class="wrap">'):]

# blocks: the header, every stepcard, every <section class="sec"> except the
# "Step by step" wrapper itself
# the header, with the run-mode box step 9 puts right under it
blocks = [("header", re.search(r"<header>.*?</header>\s*<div class=\"callout[^>]*>.*?</div>", body, re.S).group(0))]
for m in re.finditer(r'<div class="stepcard">', body):
    start = m.start()
    depth, i = 0, start
    while True:
        o = body.find("<div", i)
        c = body.find("</div>", i)
        if o != -1 and o < c:
            depth += 1
            i = o + 4
        else:
            depth -= 1
            i = c + 6
            if depth == 0:
                break
    block = body[start:i]
    title = re.sub(r"<[^>]+>", " ", re.search(r"<h3>(.*?)</h3>", block, re.S).group(1))
    blocks.append((title, block))
for m in re.finditer(r'<section class="sec">(.*?)</section>', body, re.S):
    h = re.search(r"<h2[^>]*>(.*?)</h2>", m.group(1), re.S)
    if h and "Step by step" not in h.group(1):
        blocks.append((re.sub(r"<[^>]+>", "", h.group(1)), m.group(0)))

# the segmented report: its rate-curve section (three more series) and its
# LEVEL 0 section, cut at each <h3> so every piece is one screenshot
seg = open(REPORT_SEG, encoding="utf-8").read()
seg_blocks = [("seg header", re.search(r"<header>.*?</header>\s*<div class=\"callout[^>]*>.*?</div>",
                                       seg[seg.index('<div class="wrap">'):], re.S).group(0))]
for m in re.finditer(r'<section class="sec">(.*?)</section>', seg, re.S):
    h = re.search(r"<h2[^>]*>(.*?)</h2>", m.group(1), re.S)
    if h and "exceedance-rate curve" in h.group(1):
        seg_blocks.append(("seg rate curve", m.group(0)))
    if h and h.group(1).startswith("LEVEL 0"):
        inner = m.group(1)
        cuts = [x.start() for x in re.finditer(r"<h3[^>]*>", inner)]
        edges = [0] + cuts[1:] + [len(inner)]
        for a, b in zip(edges[:-1], edges[1:]):
            t = re.search(r"<h3[^>]*>(.*?)</h3>", inner[a:b], re.S).group(1)
            seg_blocks.append(("seg " + re.sub(r"<[^>]+>", "", t),
                               f'<section class="sec">{inner[a:b]}</section>'))

tmp = tempfile.mkdtemp()
jobs = [(k, "report", t, b) for k, (t, b) in enumerate(blocks)]
jobs += [(k, "report_seg", t[4:], b) for k, (t, b) in enumerate(seg_blocks)]
for k, prefix, title, block in jobs:
    slug = re.sub(r"[^a-z0-9]+", "_", html.unescape(title).lower()).strip("_")[:40]
    page = (f"<!doctype html><meta charset='utf-8'>{style}<body>"
            f"<div class='wrap' style='padding-top:16px;padding-bottom:16px'>{block}</div></body>")
    html_path = os.path.join(tmp, f"{prefix}{k}.html")
    png_path = os.path.join(tmp, f"{prefix}{k}.png")
    open(html_path, "w", encoding="utf-8").write(page)
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    f"--screenshot={png_path}", "--window-size=1040,9000",
                    "--virtual-time-budget=20000", "file:///" + html_path.replace("\\", "/")],
                   capture_output=True)
    im = Image.open(png_path).convert("RGB")
    a = np.asarray(im).astype(int)
    bg = a[5, 5]
    rows = np.where(np.abs(a - bg).sum(axis=2).max(axis=1) > 30)[0]
    cols = np.where(np.abs(a - bg).sum(axis=2).max(axis=0) > 30)[0]
    if rows.size:
        im = im.crop((max(0, cols.min() - 8), max(0, rows.min() - 8),
                      min(im.width, cols.max() + 8), min(im.height, rows.max() + 8)))
    out = os.path.join(DOCS, "img", f"{prefix}_{k:02d}_{slug}.jpg")
    im.save(out, quality=80, optimize=True)
    print(f"{k:02d} {title.strip()[:70]:70s} {im.size} {os.path.getsize(out) // 1024} KB -> {os.path.basename(out)}")
