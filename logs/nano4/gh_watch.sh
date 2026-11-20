#!/usr/bin/env bash
# GitHub watcher for AI_CUP_2026-1 and AI_CUP_2026-2 (login node, API queries only). One stdout line per event:
#   COMMENT   a new issue/PR comment that is not this session's own ([nano4 Claude])
#   NEW       a new issue or PR
#   CHANGED   an issue/PR opened, closed or merged, or a PR with a new head commit
#   WATCH_ERROR  the API failed 5 polls in a row (once, until it recovers)
# State lives in logs/nano4/gh_watch_state, so a re-armed watcher misses nothing and repeats nothing.
# usage: gh_watch.sh [once]   The token goes to curl through a file descriptor, never on its command line.
REPOS="AI_CUP_2026-1 AI_CUP_2026-2"
S=~/BK_AICUP_2026-1/logs/nano4/gh_watch_state
mkdir -p "$S"
touch "$S/seen"
[ -f "$S/since" ] || date -u +%Y-%m-%dT%H:%M:%SZ > "$S/since"

api() {
  curl -sS --fail --max-time 30 -K <(printf 'header = "Authorization: Bearer %s"\n' "$(cat ~/.config/aicup_github_token)") \
    -H "Accept: application/vnd.github+json" "https://api.github.com/repos/bkttt0429/$1"
}

emit_once() {  # key, line
  grep -qxF "$1" "$S/seen" && return
  echo "$1" >> "$S/seen"
  echo "$2"
}

poll() {  # repo, since; returns non-zero when an API call fails
  local repo=$1 since=$2 comments items pulls first=0
  local cur="$S/items_$repo.tsv" tmp
  # the first poll records every issue and PR, so later on an unknown number really is new
  [ -f "$cur" ] || first=1
  comments=$(api "$repo/issues/comments?since=$since&per_page=100") || return 1
  items=$(api "$repo/issues?state=all&per_page=100$([ "$first" = 1 ] || echo "&since=$since")") || return 1
  pulls=$(api "$repo/pulls?state=all&sort=updated&direction=desc&per_page=100") || return 1
  while IFS=$'\t' read -r id num user at text url; do
    [ -n "$id" ] && emit_once "c$id" "COMMENT $repo#$num $user $at $text $url"
  done < <(jq -r '.[] | select(.body | startswith("[nano4 Claude]") | not)
    | [.id, (.issue_url | split("/") | last), .user.login, .created_at,
       (.body | gsub("\r"; "") | split("\n") | map(select(length > 0)) | .[0:2] | join(" ") | gsub("\t"; " ") | .[0:240]), .html_url] | @tsv' <<<"$comments")
  tmp=$(mktemp "$S/tmp.XXXXXX")
  [ -f "$cur" ] && cp "$cur" "$tmp"
  while IFS=$'\t' read -r num kind st title; do
    [ -z "$num" ] && continue
    if ! grep -q "^$num"$'\t' "$tmp"; then
      if [ "$first" = 0 ]; then emit_once "n$repo#$num" "NEW $repo#$num $kind $st $title"; fi
      printf '%s\t%s\n' "$num" "$st" >> "$tmp"
    elif ! grep -qxF "$num"$'\t'"$st" "$tmp"; then
      echo "CHANGED $repo#$num $kind -> $st $title"
      sed -i "/^$num\t/d" "$tmp"
      printf '%s\t%s\n' "$num" "$st" >> "$tmp"
    fi
  done < <(jq -r --slurpfile pulls <(printf "%s" "$pulls") '
      ($pulls[0] | map({key: (.number | tostring), value: .}) | from_entries) as $p
      | .[] | (.number | tostring) as $n
      | [$n, (if .pull_request then "PR" else "issue" end),
         (if .pull_request then
            (if $p[$n] then (if $p[$n].merged_at then "merged" else $p[$n].state end) + ":" + ($p[$n].head.sha[0:7]) else .state end)
          else .state end),
         (.title | gsub("\t"; " ") | .[0:80])] | @tsv' <<<"$items")
  mv "$tmp" "$cur"
}

fails=0
while true; do
  now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  since=$(cat "$S/since")
  ok=1
  for r in $REPOS; do poll "$r" "$since" || ok=0; done
  if [ "$ok" = 1 ]; then
    fails=0
    # two minutes of overlap: the seen list and the item states drop the repeats
    date -u -d "$now - 2 minutes" +%Y-%m-%dT%H:%M:%SZ > "$S/since"
  else
    fails=$((fails + 1))
    [ "$fails" = 5 ] && echo "WATCH_ERROR the GitHub API failed 5 polls in a row"
  fi
  [ "${1:-}" = once ] && break
  sleep 60
done
