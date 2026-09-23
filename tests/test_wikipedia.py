"""
Wikipedia parsers, on short excerpts of the real markup of each layout
(no network access).
"""

import pandas as pd

from ufc_rating.ingest import wikipedia
from conftest import make_master, raw_fight

# 2018: champion named above the table, plain (unlinked) names possible
LAYOUT_2018 = """
=== Lightweight ===
'''Champion:  {{flagicon|RUS}} [[Khabib Nurmagomedov]] [26–0]
{| class="wikitable"
!Rank
!Fighter
!Record
|-
| align="center" |1
|{{flagicon|IRL}} [[Conor McGregor]]*
| align="center" |21–3
|-
| align="center" |2
|{{flagicon|USA}} Alexander Hernandez
| align="center" |9–1
|}
"""

# 2020: champion in the table caption
LAYOUT_2020 = """
== Lightweight ==
{| class="wikitable" style="display: inline-table;"
|+
'''Champion: {{flagicon|RUS}} [[Khabib Nurmagomedov]] [28–0]'''<br>
!Rank
!Fighter
|-
| align="center" |1
|{{flagicon|USA}} [[Tony Ferguson]]
|}
"""

# 2022 onwards: champion row, interim champion, ties; 2026: two systems
LAYOUT_2026 = """
==Men's Meta rankings==
=== Light Heavyweight ===
{| class="wikitable"
! Rank
! Fighter
|-
! style="background:gold"|{{Tooltip|C|Champion}}
| {{flagicon|BRA}}
| [[Alex Pereira]]
|-
! 1
| {{flagicon|RUS}}
| [[Magomed Ankalaev]]
|}
== Men's media rankings ==
=== Men's pound-for-pound ===
{| class="wikitable"
|-
! 1
| [[Islam Makhachev]]
|}
=== Heavyweight ===
{| class="wikitable"
|-
! style="background:gold"|{{Tooltip|IC|Interim Champion}}
| {{flagicon|ENG}}
| [[Tom Aspinall]]
|-
! 6 (T)
|
| [[Nikita Krylov]]
|-
! 6 (T)
| {{flagicon|USA}}
| [[Khalil Rountree Jr.]]
|-
! 12 (T)
| {{flagicon|USA}}
| [[Anthony Smith (fighter)|Anthony Smith]]
|}
"""


def rows(text):
    return [tuple(r) for r in wikipedia.parse_rankings(text)[["system", "division", "rank", "note", "fighter"]].values]


def test_rankings_2018_layout():
    assert rows(LAYOUT_2018) == [
        ("media", "Lightweight", 0, "", "Khabib Nurmagomedov"),
        ("media", "Lightweight", 1, "", "Conor McGregor"),
        ("media", "Lightweight", 2, "", "Alexander Hernandez"),
    ]


def test_rankings_2020_caption_champion():
    assert rows(LAYOUT_2020)[0] == ("media", "Lightweight", 0, "", "Khabib Nurmagomedov")


def test_rankings_systems_interim_champions_and_ties():
    assert rows(LAYOUT_2026) == [
        ("meta", "Light Heavyweight", 0, "", "Alex Pereira"),
        ("meta", "Light Heavyweight", 1, "", "Magomed Ankalaev"),
        ("media", "Heavyweight", 0, "IC", "Tom Aspinall"),
        ("media", "Heavyweight", 6, "T", "Nikita Krylov"),
        ("media", "Heavyweight", 6, "T", "Khalil Rountree Jr."),
        ("media", "Heavyweight", 12, "T", "Anthony Smith"),
    ]


RECORD = """
== Mixed martial arts record ==
{{MMA record start}}
|-
|{{yes2}}Win
|align=center|2–1
|[[Bea Two]]
|Decision (unanimous)
|[[UFC 300]]
|{{dts|2024|April|13}}
|align=center|3
|align=center|5:00
|[[Las Vegas, Nevada]], United States
|
|-
| {{no2}}Loss
| align=center| 1–1
| Cat Three
| TKO (punches)
| Regional FC 5
| {{dts|2023|5|20}}
| align=center| 1
| align=center| 2:03
| Denver, Colorado, United States
|
|-
|{{yes2}}Win
|align=center|1–0
|Dia Four
|Submission (rear-naked choke)
|Regional FC 1
|June 1, 2022
|align=center|2
|align=center|1:10
|
|
{{end}}
"""


def test_fighter_record_table():
    record = wikipedia.parse_record(RECORD)
    assert list(record["result"]) == ["win", "loss", "win"]
    assert list(record["date"]) == [pd.Timestamp("2024-04-13"), pd.Timestamp("2023-05-20"),
                                    pd.Timestamp("2022-06-01")]
    assert record.loc[0, "opponent"] == "Bea Two"
    assert record.loc[0, "event"] == "UFC 300"
    assert record.loc[0, "location"] == "Las Vegas, Nevada, United States"
    assert record.loc[1, "method"] == "TKO (punches)"


def test_record_rows_are_matched_to_ufc_fights():
    master = make_master([raw_fight("f1", "2024-04-13", "ann one", "bea two")])
    own = wikipedia._ufc_fights(master)
    own = own[own["fighter_id"] == "ann one"]
    record = wikipedia.parse_record(RECORD)
    assert list(wikipedia.match_record(record, own)) == ["f1", None, None]


def test_record_results_written_inside_the_template():
    text = RECORD.replace("|{{yes2}}Win", "|{{yes2|Win}}", 1).replace("| {{no2}}Loss", "| {{no2|Loss}}", 1)
    assert list(wikipedia.parse_record(text)["result"]) == ["win", "loss", "win"]


def test_meta_sections_without_the_word_meta():
    # July 2026: the Meta tables came first under plain "Men's rankings" headings
    text = LAYOUT_2026.replace("==Men's Meta rankings==", "==Men's rankings==")
    assert rows(text)[0][0] == "meta" and rows(text)[2][0] == "media"
