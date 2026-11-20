"""Text-only helper: tag balance of the card files and of index.html's <main> (no browser).
  python3 reports/go_story/.review_tags.py
"""
from html.parser import HTMLParser
from pathlib import Path

D = Path(__file__).resolve().parent
VOID = {"br", "hr", "img", "input", "meta", "link", "col", "source"}


class Bal(HTMLParser):
    def __init__(self, name):
        super().__init__(); self.name, self.stack, self.bad = name, [], 0

    def handle_starttag(self, tag, attrs):
        if tag not in VOID:
            self.stack.append((tag, self.getpos()[0]))

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if self.stack and self.stack[-1][0] == tag:
            self.stack.pop(); return
        self.bad += 1
        top = self.stack[-1] if self.stack else None
        print(f"  {self.name}:{self.getpos()[0]} </{tag}> does not close {top}")
        for k in range(len(self.stack) - 1, -1, -1):   # resync
            if self.stack[k][0] == tag:
                del self.stack[k:]; break


def main():
    total = 0
    for f in sorted((D / "parts").glob("c*.html")):
        b = Bal(f.name); b.feed(f.read_text()); b.close()
        for tag, line in b.stack:
            print(f"  {f.name}:{line} <{tag}> never closed"); b.bad += 1
        print(f"{f.name}: {'OK' if not b.bad else str(b.bad) + ' problems'}")
        total += b.bad
    print("TAGS OK" if not total else f"TAGS {total} problems")


if __name__ == "__main__":
    main()
