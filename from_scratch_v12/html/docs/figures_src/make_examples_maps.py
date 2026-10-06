"""Screenshots of the "mesh on a real map" card of every example's report.html
(Esri tiles, so this needs an internet connection), cropped to the map and
saved as html/docs/img/examples_<zone>_map.jpg.

Only READS the example folders. Run from ptha18_logic_tree_test/:
    .venv/Scripts/python.exe from_scratch_v12/html/docs/figures_src/make_examples_maps.py
"""
import os
import re
import subprocess
import tempfile

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(DOCS, "..", "..", ".."))
EDGE = r"C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
FOLDERS = {"kermadectonga2_v9": "kermadectonga2", "southamerica_v9": "southamerica",
           "kurilsjapan_v9": "kurilsjapan", "makran2_v9": "makran2", "puysegur2_v9": "puysegur2",
           "calabria2_v9": "calabria2", "caribbean2_v9": "caribbean2", "antilles2_v9": "antilles2"}

tmp = tempfile.mkdtemp()
for folder, name in FOLDERS.items():
    doc = open(os.path.join(ROOT, folder, "report.html"), encoding="utf-8").read()
    style = re.search(r"<style>.*?</style>", doc, re.S).group(0)
    start = doc.index('<div id="mesh-map"')
    end = doc.index("</script>", start) + len("</script>")
    # the Leaflet <link>/<script> tags sit just before the map div
    lib = doc.rfind('<link rel="stylesheet" href="https://cdnjs', 0, start)
    page = (f"<!doctype html><meta charset='utf-8'>{style}"
            f"<body style='margin:0'><div style='width:900px;padding:10px'>{doc[lib:end]}</div></body>")
    html_path = os.path.join(tmp, f"{name}.html")
    png_path = os.path.join(tmp, f"{name}.png")
    open(html_path, "w", encoding="utf-8").write(page)
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    f"--screenshot={png_path}", "--window-size=920,760",
                    "--virtual-time-budget=20000", "file:///" + html_path.replace("\\", "/")],
                   capture_output=True)
    im = Image.open(png_path).convert("RGB").crop((10, 24, 910, 632))
    out = os.path.join(DOCS, "img", f"examples_{name}_map.jpg")
    im.save(out, quality=82, optimize=True)
    print(out, os.path.getsize(out) // 1024, "KB")
