"""Fetch diverse face photos from Wikimedia Commons (politely, cached).

Identities (public figures, freely licensed photos), 3 groups:
  east_asian:   Chloe Kim, Michelle Kwan, Yo-Yo Ma, Sandra Oh, Naomi Osaka
  south_asian:  Sundar Pichai, Satya Nadella, Virat Kohli, Priyanka Chopra, Mindy Kaling
  african:      Idris Elba, Lupita Nyong'o, Chimamanda Adichie, Naomi Campbell, Trevor Noah

Writes /tmp/bh_fairness/<group>/<person>/<n>.<ext> plus manifest.json.
"""
import json
import time
import urllib.request
from pathlib import Path

import urllib.parse

UA = "bedhead-fairness-research/0.1 (local analysis; contact dev@bedhead.local)"
OUT = Path("/tmp/bh_fairness")

GROUPS = {
    "east_asian": ["Chloe Kim", "Michelle Kwan", "Yo-Yo Ma", "Sandra Oh", "Naomi Osaka"],
    "south_asian": ["Sundar Pichai", "Satya Nadella", "Virat Kohli", "Priyanka Chopra", "Mindy Kaling"],
    "african": ["Idris Elba", "Lupita Nyong'o", "Chimamanda Ngozi Adichie", "Naomi Campbell", "Trevor Noah"],
}
WANT_PER_PERSON = 4


def api(person: str) -> list[dict]:
    cat = "Category:" + person
    url = (
        "https://commons.wikimedia.org/w/api.php?action=query&format=json"
        "&generator=categorymembers&gcmtitle=" + urllib.parse.quote(cat) +
        "&gcmtype=file&gcmlimit=20&prop=imageinfo&iiprop=url%7Cmime%7Csize"
    )
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=20) as r:
        d = json.load(r)
    out = []
    for p in d.get("query", {}).get("pages", {}).values():
        for ii in p.get("imageinfo", []):
            if ii.get("mime") in ("image/jpeg", "image/png") and ii.get("width", 0) >= 400:
                # strip tracking params from URL
                u = ii["url"].split("?")[0]
                out.append({"title": p["title"], "url": u, "mime": ii["mime"]})
    return out


def fetch(url: str, dest: Path) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            data = r.read()
    except Exception as e:
        print(f"    fetch fail {e.__class__.__name__}")
        return False
    if len(data) < 20000:  # skip tiny thumbs/decorations
        return False
    dest.write_bytes(data)
    return True


def main() -> None:
    manifest: dict[str, dict] = {}
    for group, people in GROUPS.items():
        for person in people:
            files = api(person)
            kept = 0
            entries = []
            for f in files:
                if kept >= WANT_PER_PERSON:
                    break
                ext = ".jpg" if "jpeg" in f["mime"] else ".png"
                d = OUT / group / person.replace(" ", "_").replace("'", "")
                d.mkdir(parents=True, exist_ok=True)
                dest = d / f"{kept + 1}{ext}"
                if fetch(f["url"], dest):
                    kept += 1
                    entries.append({"file": str(dest), "title": f["title"]})
                    time.sleep(0.4)  # be polite
            manifest[f"{group}/{person}"] = entries
            print(f"{group}/{person}: {kept} photos")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("manifest written:", OUT / "manifest.json")


if __name__ == "__main__":
    main()
