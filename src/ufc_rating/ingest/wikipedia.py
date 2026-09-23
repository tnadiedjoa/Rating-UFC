"""
Data from the English Wikipedia (text under CC BY-SA 4.0), read through the
MediaWiki API.

Official rankings, week by week
    The UFC publishes its rankings on ufc.com, whose terms forbid automated
    collection. The Wikipedia article "UFC rankings" reproduces them after
    every update and keeps every past version. For each week since 2018 the
    last version of the article is read and its division tables are turned
    into rows (snapshot date, system, division, rank, fighter).

    The tables changed layout several times (2018, 2020, 2022...); the
    parser only relies on what they share: one "=== Division ===" section
    per weight class, rows separated by "|-", the rank in the first cell and
    the fighter as the first link (or plain name) after the flag.

Professional records
    Fighter pages carry a "Mixed martial arts record" table with every
    professional fight, including those outside the UFC. A page is matched
    to a ufcstats fighter only when its table contains their UFC fights.
"""

import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from ufc_rating.config import WIKI_RANKINGS_CSV, WIKI_RECORDS_CSV
from ufc_rating.processing.master import NOT_A_NAME, normalize_name, parse_division

API_URL = "https://en.wikipedia.org/w/api.php"
ARTICLE = "UFC rankings"
USER_AGENT = "ufc-dataset/1.0 (https://github.com/nadiedjoa-24/Rating-UFC)"
FIRST_WEEK = "2018-01-01"   # before 2018 the article only listed the pound-for-pound top 10

RANKING_COLUMNS = ["system", "division", "rank", "note", "fighter", "page"]

# Links that are never the fighter of a row
_NOT_A_FIGHTER = re.compile(r"^(File|Image|Category|List of|UFC|Ultimate Fighting|.*\(MMA\)|"
                            r"To be determined|Mixed martial arts|Interim)", re.I)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def _api(session: requests.Session, **params) -> dict:
    params.update(format="json", formatversion=2)
    for attempt in range(4):
        try:
            response = session.get(API_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=60)
            response.raise_for_status()
            data = response.json()
            if "error" not in data:      # e.g. {"error": {"code": "maxlag"}} when the servers are busy
                return data
        except (requests.RequestException, ValueError):
            pass
        time.sleep(5 * (attempt + 1))
    raise ConnectionError(f"Wikipedia API unreachable ({params.get('action')})")


def revisions(session: requests.Session, since: str = FIRST_WEEK) -> pd.DataFrame:
    """Every revision of the article since ``since``: revid and timestamp."""
    rows, cont = [], {}
    while True:
        data = _api(session, action="query", prop="revisions", titles=ARTICLE, rvlimit=500,
                    rvprop="ids|timestamp", rvdir="newer", rvstart=f"{since}T00:00:00Z", **cont)
        rows += [{"revid": r["revid"], "timestamp": r["timestamp"]}
                 for r in data["query"]["pages"][0].get("revisions", [])]
        if "continue" not in data:
            break
        cont = data["continue"]
    out = pd.DataFrame(rows)
    out["timestamp"] = pd.to_datetime(out["timestamp"]).dt.tz_localize(None)
    return out


def weekly_revisions(revs: pd.DataFrame) -> pd.DataFrame:
    """The last revision of each week (Monday to Sunday), with its ``week`` (the Monday)."""
    revs = revs.sort_values("timestamp").copy()
    revs["week"] = revs["timestamp"].dt.to_period("W-SUN").dt.start_time
    return revs.groupby("week").tail(1).reset_index(drop=True)


def resolve_titles(session: requests.Session, titles, pause: float = 1.0) -> dict:
    """{title: title of the page it leads to}, following redirects (50 titles per request)."""
    titles = list(titles)
    resolved = {t: t for t in titles}
    for start in range(0, len(titles), 50):
        batch = titles[start:start + 50]
        data = _api(session, action="query", titles="|".join(batch), redirects=1)["query"]
        for step in data.get("normalized", []) + data.get("redirects", []):
            for title, current in resolved.items():
                if current == step["from"]:
                    resolved[title] = step["to"]
        time.sleep(pause)
    return resolved


def wikitext(session: requests.Session, revid: int) -> str:
    data = _api(session, action="query", prop="revisions", revids=revid,
                rvprop="content", rvslots="main")
    return data["query"]["pages"][0]["revisions"][0]["slots"]["main"]["content"]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    text = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", "", text, flags=re.S)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"<[^>]+>|'{2,}|&nbsp;", " ", text)
    return text.strip(" |!\n")


def _plain(text: str) -> str:
    """Wiki markup -> plain text: '[[UFC 330]]' -> 'UFC 330', '[[A|B]], C' -> 'B, C'."""
    text = re.sub(r"\[\[(?:[^\]|]+\|)?([^\]]+)\]\]", r"\1", text)
    return re.sub(r"\s+", " ", _clean(text)).strip()


def _tidy_name(name: str) -> str:
    """'Tony Ferguson*' -> 'Tony Ferguson'; 'Kevin Lee (fighter)' -> 'Kevin Lee'."""
    name = re.sub(r"[*†‡]", "", name)
    return re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()


def _fighter_link(cells: list):
    """(name, linked page) of the first link in ``cells``, or (plain name, None); (None, None) if none."""
    for cell in cells:
        for target, label in re.findall(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", cell):
            if not _NOT_A_FIGHTER.match(target.strip()):
                page = target.split("#")[0].strip()
                return _tidy_name(label or target), page[:1].upper() + page[1:]
        plain = _clean(cell)
        if re.fullmatch(r"[^\d\[\]{}=<>|]{3,}", plain) and "align" not in plain:
            return _tidy_name(plain), None
    return None, None


def _fighter(cells: list) -> Optional[str]:
    """First link (display text) or plain name in ``cells``."""
    return _fighter_link(cells)[0]


def _rank(cell: str):
    """
    (rank, note) from the first cell of a row, or None for other rows.
    Cells look like '! 5', '| align="center" |5', '! 6 (T)' (tie) or
    '! style="background:gold"|{{Tooltip|C|Champion}}'. Champions and
    interim champions get rank 0, with note 'IC' for the latter.
    """
    cell = re.sub(r"\{\{Tooltip\|([^|}]+)\|[^}]*\}\}", r"\1", cell)
    value = _clean(cell.split("|")[-1])
    match = re.fullmatch(r"(\d{1,2})\s*(\(T\)|T)?", value)
    if match:
        return int(match.group(1)), "T" if match.group(2) else ""
    return {"C": (0, ""), "IC": (0, "IC")}.get(value)


def parse_rankings(text: str) -> pd.DataFrame:
    """
    Division tables of one version of the article ->
    (system, division, rank, note, fighter).

    ``system`` is 'media' for the rankings voted by a media panel (2013 to
    2026) and 'meta' for the model-based Meta UFC Rankings that replaced
    them from June 2026 (both were published during the transition).
    ``note``: 'T' for a tied rank, 'IC' for an interim champion (rank 0).
    ``page``: the Wikipedia page the name links to (None for plain names).
    """
    rows = []
    system = "media"
    parts = re.split(r"^(==+[^=\n]+==+)\s*$", text, flags=re.M)
    # When a "media rankings" section exists, the other ranking sections are the
    # Meta ones, even when their heading is just "Men's rankings" (July 2026)
    has_media_section = any("media" in h.lower() for h in parts[1::2] if not h.startswith("==="))
    for header, body in zip(parts[1::2], parts[2::2]):
        title = header.strip("= ")
        division = parse_division(title)
        if division is None:
            if not header.startswith("==="):          # a level-2 heading starts a new system
                lowered = title.lower()
                if "media" in lowered:
                    system = "media"
                elif "meta" in lowered or (has_media_section and "ranking" in lowered):
                    system = "meta"
                else:
                    system = "media"
            continue
        if "pound" in title.lower() or "{|" not in body:
            continue
        table = body[body.index("{|"):]
        header_area = body[:body.index("{|")] + table.split("\n|-", 1)[0]
        found = []
        # Older layouts name the champion above the table or in its caption
        for interim, line in re.findall(r"(Interim\s+)?Champion\s*:'*\s*(.+)", header_area):
            name, page = _fighter_link([line])
            if name:
                found.append((0, "IC" if interim else "", name, page))
        for row in re.split(r"\n\|-[^\n]*", table)[1:]:
            lines = [l for l in row.split("\n") if l.strip() and not l.startswith("|}")]
            cells = [c for line in lines for c in re.split(r"\|\||!!", line)]
            rank = _rank(cells[0]) if cells else None
            name, page = _fighter_link(cells[1:]) if rank else (None, None)
            if name and name.lower() not in NOT_A_NAME:
                found.append((*rank, name, page))
        seen = set()
        for rank, note, name, page in found:
            if (rank, name) not in seen:
                seen.add((rank, name))
                rows.append({"system": system, "division": division, "rank": rank,
                             "note": note, "fighter": name, "page": page})
    return pd.DataFrame(rows, columns=RANKING_COLUMNS)


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

def rankings_history(out_path: Path = WIKI_RANKINGS_CSV, since: str = FIRST_WEEK,
                     pause: float = 1.0) -> pd.DataFrame:
    """
    Weekly snapshots of the official rankings, appended to ``out_path``.

    Each snapshot is the last version of the article in its week, dated by
    that version's day (the rankings it shows were current on that day).
    Versions already in the file are not downloaded again; when the current
    week has a newer version, it replaces that week's snapshot.
    """
    session = requests.Session()
    known = pd.read_csv(out_path, parse_dates=["date"]) if Path(out_path).exists() else pd.DataFrame()
    weeks = weekly_revisions(revisions(session, since))
    if not known.empty:
        weeks = weeks[~weeks["revid"].isin(known["revid"])]
        stale = known["date"].dt.to_period("W-SUN").dt.start_time.isin(weeks["week"])
        known = known[~stale]
    frames = [known]
    for week in weeks.itertuples():
        table = parse_rankings(wikitext(session, week.revid))
        table.insert(0, "date", week.timestamp.normalize())
        table["revid"] = week.revid
        frames.append(table)
        time.sleep(pause)
    history = pd.concat(frames, ignore_index=True).sort_values(["date", "system", "division", "rank"])
    history["page"] = history["page"].map(resolve_titles(session, history["page"].dropna().unique()))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(out_path, index=False)
    return history


# ---------------------------------------------------------------------------
# Fighter records (every professional fight, inside and outside the UFC)
# ---------------------------------------------------------------------------

_RESULTS = {"win": "win", "loss": "loss", "draw": "draw", "nc": "nc", "no contest": "nc"}
_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}
_RECORD_COLUMNS = ["result", "record", "opponent", "method", "event", "date", "round", "time", "location"]


def pages(session: requests.Session, titles: list) -> dict:
    """{requested title: (page title, wikitext)} for the titles that exist (at most 50)."""
    data = _api(session, action="query", prop="revisions", rvprop="content", rvslots="main",
                titles="|".join(titles), redirects=1)["query"]
    alias = {t: t for t in titles}
    for step in data.get("normalized", []) + data.get("redirects", []):
        for requested, current in list(alias.items()):
            if current == step["from"]:
                alias[requested] = step["to"]
    texts = {p["title"]: p["revisions"][0]["slots"]["main"]["content"]
             for p in data["pages"] if "revisions" in p}
    # A response over the size limit leaves some existing pages without content: fetch them alone
    for page in data["pages"]:
        if "revisions" not in page and "missing" not in page and "invalid" not in page and len(titles) > 1:
            texts.update({t: text for t, text in pages(session, [page["title"]]).values()})
    return {requested: (title, texts[title]) for requested, title in alias.items() if title in texts}


def _date(cell: str) -> Optional[pd.Timestamp]:
    """{{dts|2002|September|28}}, {{dts|2025|11|02}}, 'August 15, 2026', '15 August 2026'..."""
    dts = re.search(r"\{\{\s*dts\s*\|([^}]*)\}\}", cell, flags=re.I)
    text = dts.group(1) if dts else _clean(cell)
    parts = [p.strip() for p in text.split("|") if p.strip() and "=" not in p]
    if len(parts) >= 3 and parts[0].isdigit():
        month = int(parts[1]) if parts[1].isdigit() else _MONTHS.get(parts[1].lower())
        if month and parts[2].isdigit():
            return pd.Timestamp(int(parts[0]), month, int(parts[2]))
    parsed = pd.to_datetime(" ".join(parts), errors="coerce")
    return None if pd.isna(parsed) else parsed


def parse_record(text: str) -> pd.DataFrame:
    """The "Mixed martial arts record" table of a fighter page, one row per fight."""
    start = text.find("{{MMA record start")
    if start < 0:
        return pd.DataFrame(columns=_RECORD_COLUMNS)
    end = text.find("{{end}}", start)
    table = text[start:end if end > 0 else None]
    rows = []
    for chunk in re.split(r"\n\|-[^\n]*", table)[1:]:
        cells = []
        for line in chunk.split("\n"):
            if line.startswith("|") and not line.startswith("|}"):
                cells += line[1:].split("||")
        cells = [re.sub(r"^\s*(align|style|rowspan)\s*=[^|]*\|", "", c).strip() for c in cells]
        if len(cells) < 6:
            continue
        # '{{yes2}}Win', '{{yes2|Win}}', 'Win'...
        outcome = re.search(r"\b(win|loss|draw|nc|no contest)\b", cells[0], flags=re.I)
        result = _RESULTS.get(outcome.group(1).lower()) if outcome else None
        date = _date(cells[5])
        if result is None or date is None:
            continue
        extra = [_plain(c) for c in cells[6:9]] + [None] * (3 - len(cells[6:9]))
        rows.append(dict(zip(_RECORD_COLUMNS, [
            result, _plain(cells[1]), _fighter([cells[2]]) or _plain(cells[2]),
            _plain(cells[3]), _plain(cells[4]), date, *extra])))
    return pd.DataFrame(rows, columns=_RECORD_COLUMNS)


def _ufc_fights(master: pd.DataFrame) -> pd.DataFrame:
    """One row per (fighter, UFC fight): id, name, date, normalised opponent name."""
    parts = []
    for side, opp in (("r", "b"), ("b", "r")):
        parts.append(pd.DataFrame({"fighter_id": master[f"{side}_id"], "name": master[f"{side}_name"],
                                   "date": master["date"], "fight_id": master["fight_id"],
                                   "opponent": master[f"{opp}_name"].map(normalize_name)}))
    return pd.concat(parts, ignore_index=True)


def match_record(record: pd.DataFrame, fights: pd.DataFrame, max_days: int = 2) -> pd.Series:
    """
    ufcstats fight_id of each row of a Wikipedia record (None for fights
    outside the UFC): same opponent (normalised full name or last name)
    within ``max_days``.
    """
    ids = []
    last_names = fights["opponent"].str.split().str[-1]
    for row in record.itertuples():
        near = (fights["date"] - row.date).abs().dt.days <= max_days
        opponent = normalize_name(row.opponent)
        last = opponent.split()[-1] if opponent else ""
        same = fights[near & ((fights["opponent"] == opponent) | (last_names == last))]
        ids.append(same["fight_id"].iloc[0] if len(same) else None)
    return pd.Series(ids, index=record.index, dtype=object)


def fighter_records(master: pd.DataFrame, fighter_ids=None, pause: float = 1.0,
                    min_share: float = 0.5) -> pd.DataFrame:
    """
    Professional record of the fighters of ``master`` (all, or
    ``fighter_ids``) who have a Wikipedia page with a "Mixed martial arts
    record" table.

    Candidate pages are "<name>" and "<name> (fighter)". A page is kept when
    at least ``min_share`` of the fighter's UFC fights are found in its table
    (same opponent within two days), which rules out namesakes. Each row
    carries ``fight_id`` when it is one of our UFC fights, None otherwise.
    """
    session = requests.Session()
    fights = _ufc_fights(master)
    names = fights.drop_duplicates("fighter_id").set_index("fighter_id")["name"]
    if fighter_ids is not None:
        names = names[names.index.isin(set(fighter_ids))]
    candidates = {fid: [name, f"{name} (fighter)"] for fid, name in names.items()}
    titles = sorted({t for pair in candidates.values() for t in pair})

    texts = {}
    for start in range(0, len(titles), 50):
        texts.update(pages(session, titles[start:start + 50]))
        time.sleep(pause)

    by_fighter = dict(tuple(fights.groupby("fighter_id")))
    frames = []
    for fid, options in candidates.items():
        own = by_fighter[fid]
        best = None
        for title in options:
            if title not in texts:
                continue
            page_title, text = texts[title]
            record = parse_record(text)
            if record.empty:
                continue
            record["fight_id"] = match_record(record, own)
            share = record["fight_id"].nunique() / own["fight_id"].nunique()
            if share >= min_share and (best is None or share > best[0]):
                best = (share, page_title, record)
        if best:
            record = best[2]
            record.insert(0, "fighter_id", fid)
            record.insert(1, "wiki_title", best[1])
            frames.append(record)
    columns = ["fighter_id", "wiki_title"] + _RECORD_COLUMNS + ["fight_id"]
    return pd.concat(frames, ignore_index=True)[columns] if frames else pd.DataFrame(columns=columns)


def update_records(master: pd.DataFrame, out_path: Path = WIKI_RECORDS_CSV, pause: float = 1.0,
                   lookback_days: int = 60) -> pd.DataFrame:
    """
    Keep ``out_path`` up to date: the first run reads every fighter; later
    runs only the fighters with a UFC fight in the ``lookback_days`` before
    the last run or after it (their record changed, or they are new; the
    look-back covers events that reach the data late, through the Kaggle
    mirror). ``fetched`` holds the day each record was read.
    """
    known = pd.read_csv(out_path, parse_dates=["date", "fetched"]) if Path(out_path).exists() else None
    if known is None or known.empty:
        ids = None
    else:
        last_run = known["fetched"].max()
        recent = master[master["date"] >= last_run - pd.Timedelta(days=lookback_days)]
        ids = set(recent["r_id"]) | set(recent["b_id"])
    fresh = fighter_records(master, ids, pause=pause)
    fresh["fetched"] = pd.Timestamp.today().normalize()
    if known is not None and ids is not None:
        known = known[~known["fighter_id"].isin(ids)]
        fresh = pd.concat([known, fresh], ignore_index=True)
    fresh = fresh.sort_values(["fighter_id", "date"]).reset_index(drop=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fresh.to_csv(out_path, index=False)
    return fresh
