"""Fetch more diverse face photos from Wikimedia Commons (10/person target)."""
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

UA = "bedhead-fairness-research/0.1 (local analysis; contact dev@bedhead.local)"
OUT = Path("/tmp/bh_fairness")

GROUPS = {
    "east_asian": ["Chloe Kim", "Michelle Kwan", "Yo-Yo Ma", "Sandra Oh", "Naomi Osaka", "Liu Yifei", "Fan Bingbing"],
    "south_asian": ["Sundar Pichai", "Satya Nadella", "Virat Kohli", "Priyanka Chopra", "Mindy Kaling", "Ravi Ashwin", "Mira Nair"],
    "african": ["Idris Elba", "Lupita Nyong'o", "Chimamanda Ngozi Adichie", "Naomi Campbell", "Trevor Noah", "Didier Drogba", "Angelique Kidjo"],
}
WANT = 10


def api(person):
    cat = "Category:" + person
    url = ("https://commons.wikimedia.org/w/api.php?action=query&format=json"
           "&generator=categorymembers&gcmtitle=" + urllib.parse.quote(cat) +
           "&gcmtype=file&gcmlimit=40&prop=imageinfo&iiprop=url%7Cmime%7Csize")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=20) as r:
        d = json.load(r)
    out = []
    for p in d.get("query", {}).get("pages", {}).values():
        for ii in p.get("imageinfo", []):
            if ii.get("mime") in ("image/jpeg", "image/png") and ii.get("width", 0) >= 600:
                out.append({"url": ii["url"].split("?")[0], "mime": ii["mime"]})
    return out


def fetch(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            data = r.read()
    except (OSError, urllib.error.URLError):
        return False
    if len(data) < 40000:
        return False
    dest.write_bytes(data)
    return True


manifest = {}
for group, people in GROUPS.items():
    for person in people:
        files = api(person)
        kept, entries = 0, []
        for f in files:
            if kept >= WANT:
                break
            ext = ".jpg" if "jpeg" in f["mime"] else ".png"
            d = OUT / group / person.replace(" ", "_").replace("'", "")
            d.mkdir(parents=True, exist_ok=True)
            dest = d / f"{kept + 1}{ext}"
            if dest.exists():
                kept += 1
                continue
            if fetch(f["url"], dest):
                kept += 1
                entries.append({"file": str(dest)})
                time.sleep(0.3)
        manifest[f"{group}/{person}"] = entries
        print(f"{group}/{person}: {kept}")
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
print("done")
