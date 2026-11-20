"""Poll bkttt0429/AI_CUP_2026-1 for new issues, PRs and comments not written by nano4 Claude (not in git).

One stdout line per new item; the state file keeps seen ids across re-arms. --seed marks everything present as seen
without printing. The loop exits after MAX_SECONDS so an orphaned poller never eats items its reader will not see.
"""
import json, os, sys, time, urllib.request, urllib.error

REPO = "bkttt0429/AI_CUP_2026-1"
API = f"https://api.github.com/repos/{REPO}"
STATE = sys.argv[1]
INTERVAL = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 60
TOKEN = open(os.path.expanduser("~/.config/aicup_github_token")).read().strip()
MINE = "[nano4 Claude]"
MAX_SECONDS = 29 * 60


def get(path):
    req = urllib.request.Request(API + path, headers={"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def load():
    try:
        return json.load(open(STATE))
    except (OSError, ValueError):
        return {"seen": [], "since": None}


def snippet(body):
    return " ".join((body or "").split())[:160]


def poll(state):
    seen = set(state["seen"])
    since = state["since"] or "2026-10-01T00:00:00Z"
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 120))
    out = []
    for it in get(f"/issues?state=all&since={since}&per_page=100"):
        key = f"issue:{it['id']}"
        if key not in seen:
            seen.add(key)
            if not (it.get("body") or "").lstrip().startswith(MINE):
                out.append(f"new {'PR' if 'pull_request' in it else 'issue'} #{it['number']} \"{it['title']}\": {snippet(it.get('body'))}")
    for c in get(f"/issues/comments?since={since}&per_page=100"):
        key = f"comment:{c['id']}"
        if key not in seen:
            seen.add(key)
            if not (c.get("body") or "").lstrip().startswith(MINE):
                out.append(f"comment on #{c['issue_url'].rsplit('/', 1)[-1]} (id {c['id']}): {snippet(c.get('body'))}")
    for c in get(f"/pulls/comments?since={since}&per_page=100"):
        key = f"review:{c['id']}"
        if key not in seen:
            seen.add(key)
            if not (c.get("body") or "").lstrip().startswith(MINE):
                out.append(f"review comment on PR #{c['pull_request_url'].rsplit('/', 1)[-1]} {c.get('path')}: {snippet(c.get('body'))}")
    state["seen"], state["since"] = sorted(seen), now
    json.dump(state, open(STATE, "w"))
    return out


def main():
    state = load()
    if "--seed" in sys.argv:
        poll(state)
        return
    fails, deadline = 0, time.time() + MAX_SECONDS
    while time.time() < deadline:
        try:
            for line in poll(state):
                print(line, flush=True)
            fails = 0
        except (urllib.error.URLError, OSError, ValueError) as e:
            fails += 1
            print(f"poll failed: {e}", file=sys.stderr, flush=True)
            if fails == 3:
                print(f"WATCH ERROR: 3 consecutive poll failures, last: {e}", flush=True)
        time.sleep(INTERVAL)


main()
