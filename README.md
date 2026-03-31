# Animal Kingdom Data Generator

Generate synthetic data for 5 related PostgreSQL tables about the animal kingdom using the **Groq API** (`openai/gpt-oss-120b`).

## Tables

| Table | Rows | Description |
|-------|------|-------------|
| `habitats` | 20 | Ecosystems / biomes |
| `species` | 200 | Animal species linked to habitats |
| `animals` | 1,000 | Individual animals linked to species |
| `diet_logs` | 1,000 | Feeding records linked to animals |
| `observations` | 1,000 | Wildlife observations linked to animals |

**Joins:** `habitats` ← `species` ← `animals` ← `diet_logs` / `observations`

## Prerequisites

- **Python 3.10+**
- **PostgreSQL** running locally (or remote)
- **Groq API key** — get one at [console.groq.com](https://console.groq.com)

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create your .env file
cp .env.example .env
# Edit .env with your GROQ_API_KEY and DATABASE_URL

# 3. Create the database (if it doesn't exist)
createdb animal_kingdom

# 4. Run the generator
python generate_data.py
```

## Rate Limits

The script respects the `openai/gpt-oss-120b` rate limits:

| Limit | Value | How it's handled |
|-------|-------|-----------------|
| RPM | 30 | 2.5s sleep between API calls |
| RPD | 1,000 | Batch generation (~70 total calls) |
| TPM | 8,000 | Concise JSON-only prompts |
| TPD | 200,000 | Batched rows (25–50 per call) |

Automatic retry with exponential backoff on failures.

## Verification

After running, the script prints:
- Row counts per table
- Sample JOIN query results (3-table and 4-table joins)
- API usage statistics (requests, tokens, elapsed time)

```sqlz
-- Manual verification
SELECT a.name, s.common_name, h.name AS habitat
FROM animals a
JOIN species s ON a.species_id = s.id
JOIN habitats h ON s.habitat_id = h.id
LIMIT 10;
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Your Groq API key | — (required) |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://postgres:postgres@localhost:5432/animal_kingdom` |


"Bengal Tiger belongs to which habitat?"
"What does the Bald Eagle eat?"
"How many Blue Whales are there?"
"Which habitat does the Snow Leopard live in?"
"Show me all observations of the King Cobra"

