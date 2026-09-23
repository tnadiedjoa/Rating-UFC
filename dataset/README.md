# UFC dataset

Every UFC fight from 12 November 1993 to 19 September 2026: 8,905 fights, 20,904 rounds and 2,760 fighters, with results, round-by-round statistics, judges' scores, bonuses, betting odds and official rankings. Built and updated weekly by the code of this repository.

## Files

| File | Rows | One row per |
|---|---:|---|
| `fights.csv` | 8,905 | UFC fight |
| `rounds.csv` | 20,904 | round of a fight with statistics |
| `fighters.csv` | 2,760 | fighter with at least one UFC fight |
| `elo.csv` | 17,810 | fighter in a fight (Elo before and after) |
| `rankings.csv` | 77,503 | fighter in a weekly official ranking (January 2018 onwards) |
| `records.csv` | 46,223 | professional fight of a fighter with a Wikipedia record (1,791 fighters) |

All files are UTF-8 CSV. Fighters, fights and events keep the ids of ufcstats.com, so the tables join on `fighter_id`, `fight_id` and `event_id`.

## Coverage

| Field | Available for |
|---|---|
| Fight totals (strikes, takedowns...) | 100.0% of fights |
| Control time | 97.6% of fights (not recorded before mid-1999) |
| Round-by-round statistics | 99.8% of fights (some 1990s tournament fights only have totals) |
| Judges' scores | 97.2% of decisions (draws have scores but no orientation, see below) |
| Betting odds | 94.2% of fights since 2010 |
| Official rank at fight time | fights since 2010 (ranked fighters only) |
| Reach | 76.2% of fighters |
| Date of birth | 96.3% of fighters |
| Professional record (Wikipedia) | 64.9% of fighters |

## Things to know

- **Corners.** `r_` is the fighter listed first on ufcstats.com. Since about 2010 this is the red corner; before, ufcstats lists the **winner first in every fight**, so the r/b sides must not be used as corners for those years (a model learning from them would learn the result).
- **Outcome.** `outcome` is `r`, `b`, `draw` or `nc` (no contest); `winner_id` is empty for draws and no contests.
- **Judges' scores** are oriented to the r/b sides using the winner (ufcstats writes them as loser - winner). They are left empty for draws, which cannot be oriented.
- **Odds** are American odds, closing lines: the Ultimate UFC Dataset (to March 2026), completed by the median over the sportsbooks listed on bestfightodds.com for the fights it misses since 2023 and for every later event (`odds_source`).
- **Official ranks** are 0 for the champion (and an interim champion), empty when unranked. `r_rank`/`b_rank` are the media-panel rankings; `r_meta_rank`/`b_meta_rank` the Meta UFC Rankings that replaced them from June 2026. In `rankings.csv`, `note` is `T` for a tied rank and `IC` for an interim champion.
- **Bonuses** are listed per fight. Fight of the Night goes to both fighters; the other bonuses usually go to the winner.
- **Elo** uses K = 80, with split and majority decisions counting half (tuned on fights before the models' test period).

## Sources and licence

| Data | Source | Licence |
|---|---|---|
| Fights, rounds, fighter profiles, judges, bonuses | [ufcstats.com](http://ufcstats.com), through the [UFC Datasets 1994-2025](https://www.kaggle.com/datasets/neelagiriaditya/ufc-datasets-1994-2025) mirror and this repository's scraper for the latest events | CC0 (mirror) |
| Odds up to March 2026, ranks up to 2018 | [Ultimate UFC Dataset](https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset) | CC BY 4.0 |
| Odds it misses since 2023, later odds | [bestfightodds.com](https://www.bestfightodds.com) (sportsbook consensus) | facts, source credited |
| Official rankings, professional records | [English Wikipedia](https://en.wikipedia.org/wiki/UFC_rankings) | CC BY-SA 4.0 |

Because it includes material from Wikipedia, the dataset as a whole is distributed under **CC BY-SA 4.0**: credit the sources above and share adaptations under the same licence.
