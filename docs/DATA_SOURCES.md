# Ball Knower – Data Sources & Naming Standard

This document defines the category-first filename convention and all expected weekly datasets.

This standard replaces older provider-based filenames (e.g., `nfelo_power_ratings_*`, `substack_qb_epa_*`).
Providers may vary, but filenames remain consistent.

---

# 1. Naming Convention (Mandatory)

All weekly source files must follow:

```
{category}_{season}week{week}.csv
```

Where:
- `{category}` is one of the approved data categories
- `{season}` is a 4-digit year
- `{week}` is an integer (1–22)

Example:

```
epa_team_2025_week_11.csv
power_team_2025_week_11.csv
qb_epa_2025_week_11.csv
schedule_2025_week_11.csv
```

---

# 2. Canonical Categories & Descriptions

| Category Name     | Purpose / Contents |
|-------------------|--------------------|
| `schedule`        | Weekly matchups, home/away teams, kickoff times, stadium info |
| `power_team`      | Team-level rating systems (composite blends of nfelo, 538, Substack, etc.) |
| `epa_team`        | Offensive/defensive EPA per play, efficiency splits |
| `epa_player`      | Player-level EPA metrics |
| `qb_epa`          | Quarterback EPA per play, composite rankings |
| `qb_ratings`      | QB grading systems (QBR, Elo-like ratings, efficiency tiers) |
| `injuries`        | Player injury reports, statuses, in/out flags |
| `weather`         | Forecasted game weather (temp, wind, precipitation) |

---

# 3. Provider-Agnostic Data Architecture

Providers (nfelo, Substack, 538, custom blends) no longer appear in filenames.
Instead, multiple models feeding the same category are blended inside the data file.

Example:
`power_team_2025_week_11.csv` may include:

- `rating_nfelo`
- `rating_ss_prime`
- `rating_538_elo`
- `rating_blend_primary`

Similarly, `epa_team_2025_week_11.csv` may include:

- `off_epa`
- `def_epa`
- `off_success_rate`
- `def_success_rate`

---

# 4. Expected Weekly Files

Each week must include exactly one CSV per category—for example:

```
schedule_2025_week_11.csv
power_team_2025_week_11.csv
epa_team_2025_week_11.csv
epa_player_2025_week_11.csv
qb_epa_2025_week_11.csv
qb_ratings_2025_week_11.csv
injuries_2025_week_11.csv
weather_2025_week_11.csv
```

---

# 5. Category → Example Filename Mapping

| Category       | Example Filename |
|----------------|------------------|
| schedule       | schedule_2025_week_11.csv |
| power_team     | power_team_2025_week_11.csv |
| epa_team       | epa_team_2025_week_11.csv |
| epa_player     | epa_player_2025_week_11.csv |
| qb_epa         | qb_epa_2025_week_11.csv |
| qb_ratings     | qb_ratings_2025_week_11.csv |
| injuries       | injuries_2025_week_11.csv |
| weather        | weather_2025_week_11.csv |

---

# 6. Legacy Filename Support

The unified loader continues to support older filenames such as:

- `nfelo_power_ratings_2025_week_11.csv`
- `substack_qb_epa_2025_week_11.csv`

…but only as a fallback.
All new data must use the category-first standard.

---

End of document.
