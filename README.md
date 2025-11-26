# CourtIQ – Computer Vision Backend (v0)

This is the backend for **CourtIQ**, handling real video analysis:

- Accepts a game film upload
- Uses YOLOv8 to detect players
- Filters by the jersey color selected by the coach
- Builds a rough per-player usage/profile report
- Returns JSON that the Base44 frontend can render

## API

### `POST /analyze`

**Form-data:**

- `file` (required) — video file (MP4, MOV, etc.)
- `jersey_color` — string, e.g. `"royal-blue"`, `"red"`, `"green"`, etc.
- `game_title` — string
- `opponent` — string

**Response:**

```json
{
  "game_id": "uuid...",
  "report": {
    "video_path": "...",
    "jersey_color_analyzed": "royal-blue",
    "generated_at": "2025-11-25T20:00:00Z",
    "team_summary": { ... },
    "players": [ ... ],
    "plays": [],
    "defense": {},
    "notes_for_coach": [ ... ],
    "game_title": "Your title",
    "opponent": "Opponent name"
  }
}
