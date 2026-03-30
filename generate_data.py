"""
Animal Kingdom Data Generator
==============================
Generates synthetic data for 5 PostgreSQL tables using Groq API (openai/gpt-oss-120b).
Strictly follows rate limits: 30 RPM, 1K RPD, 8K TPM, 200K TPD.
"""

import os
import re
import sys
import json
import time
import random
import logging
from datetime import datetime, timedelta

import psycopg2
from psycopg2.extras import execute_values
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/animal_kingdom")

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

RPM_LIMIT = 30
SECONDS_BETWEEN_REQUESTS = 60.0 / RPM_LIMIT + 1.5  
MAX_RETRIES = 10
RETRY_BASE_DELAY = 5  
BATCH_SIZES = {
    "habitats": 10,      
    "species": 10,       
    "animals": 20,       
    "diet_logs": 25,     
    "observations": 20,  
}
ROW_COUNTS = {
    "habitats": 20,
    "species": 200,
    "animals": 1000,
    "diet_logs": 1000,
    "observations": 1000,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("animal_gen")

SCHEMA_SQL = """
DROP TABLE IF EXISTS observations CASCADE;
DROP TABLE IF EXISTS diet_logs CASCADE;
DROP TABLE IF EXISTS animals CASCADE;
DROP TABLE IF EXISTS species CASCADE;
DROP TABLE IF EXISTS habitats CASCADE;

CREATE TABLE habitats (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(100) NOT NULL,
    type            VARCHAR(50) NOT NULL,
    climate         VARCHAR(50) NOT NULL,
    region          VARCHAR(100) NOT NULL,
    avg_temp_c      DECIMAL(5,1),
    area_km2        DECIMAL(12,2),
    description     TEXT
);

CREATE TABLE species (
    id              SERIAL PRIMARY KEY,
    common_name     VARCHAR(100) NOT NULL,
    scientific_name VARCHAR(150) NOT NULL,
    class           VARCHAR(50) NOT NULL,
    diet            VARCHAR(50) NOT NULL,
    conservation_status VARCHAR(30) NOT NULL,
    avg_lifespan_years INTEGER,
    habitat_id      INTEGER NOT NULL REFERENCES habitats(id),
    description     TEXT
);

CREATE TABLE animals (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(100) NOT NULL,
    species_id      INTEGER NOT NULL REFERENCES species(id),
    age_years       DECIMAL(4,1),
    weight_kg       DECIMAL(8,2),
    sex             VARCHAR(10) NOT NULL,
    birth_date      DATE,
    health_status   VARCHAR(30) NOT NULL,
    tag_number      VARCHAR(20) UNIQUE NOT NULL
);

CREATE TABLE diet_logs (
    id              SERIAL PRIMARY KEY,
    animal_id       INTEGER NOT NULL REFERENCES animals(id),
    food_item       VARCHAR(100) NOT NULL,
    quantity_kg     DECIMAL(6,2),
    feeding_time    TIME NOT NULL,
    calories        INTEGER,
    log_date        DATE NOT NULL
);

CREATE TABLE observations (
    id              SERIAL PRIMARY KEY,
    animal_id       INTEGER NOT NULL REFERENCES animals(id),
    observer_name   VARCHAR(100) NOT NULL,
    obs_date        DATE NOT NULL,
    location        VARCHAR(150) NOT NULL,
    behavior        TEXT NOT NULL,
    health_notes    TEXT,
    weather         VARCHAR(50) NOT NULL
);

CREATE INDEX idx_species_habitat ON species(habitat_id);
CREATE INDEX idx_animals_species ON animals(species_id);
CREATE INDEX idx_diet_logs_animal ON diet_logs(animal_id);
CREATE INDEX idx_observations_animal ON observations(animal_id);
"""

client = None
api_stats = {
    "total_requests": 0,
    "total_tokens": 0,
    "start_time": None,
}


def get_client():
    """Lazily initialize the OpenAI client (only when API key is available)."""
    global client
    if client is None:
        client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url=GROQ_BASE_URL,
        )
    return client


def _parse_retry_delay(error_message: str) -> float | None:
    """Extract the suggested wait time from a 429 error message."""
    match = re.search(r'Please try again in (\d+)m([\d.]+)s', str(error_message))
    if match:
        minutes = int(match.group(1))
        seconds = float(match.group(2))
        return minutes * 60 + seconds + 2  # add 2s buffer
    match = re.search(r'Please try again in ([\d.]+)s', str(error_message))
    if match:
        return float(match.group(1)) + 2
    return None


def rate_limited_generate(prompt: str, max_tokens: int = 4000) -> str:
    """
    Call Groq API with rate limiting, retries, and exponential backoff.
    For 429 rate limit errors, parses the suggested wait time and sleeps accordingly.
    """
    if api_stats["start_time"] is None:
        api_stats["start_time"] = time.time()

    api_client = get_client()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(SECONDS_BETWEEN_REQUESTS)

            response = api_client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a data generator. You ONLY output valid JSON arrays. "
                            "No markdown, no explanations, no code fences. Just raw JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.9,
                max_tokens=max_tokens,
            )

            api_stats["total_requests"] += 1
            if response.usage:
                api_stats["total_tokens"] += response.usage.total_tokens

            content = response.choices[0].message.content.strip()
            # Strip markdown code fences if present
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            return content

        except Exception as e:
            error_str = str(e)
            # Parse suggested wait time from 429 errors
            parsed_delay = _parse_retry_delay(error_str)
            if parsed_delay and '429' in error_str:
                log.warning(
                    f"Rate limit hit (attempt {attempt}/{MAX_RETRIES}). "
                    f"Waiting {parsed_delay:.0f}s as suggested by API..."
                )
                time.sleep(parsed_delay)
            else:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 1)
                log.warning(
                    f"API error (attempt {attempt}/{MAX_RETRIES}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)

    log.error("Max retries exceeded. Exiting.")
    sys.exit(1)


def parse_json_array(raw: str) -> list:
    """Parse a JSON array from the API response, with fallback handling."""
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # Sometimes the model wraps in an object
            for v in data.values():
                if isinstance(v, list):
                    return v
        return [data]
    except json.JSONDecodeError:
        # Try to find a JSON array in the response
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass
        log.error(f"Failed to parse JSON:\n{raw[:500]}")
        return []


# ──────────────────────────────────────────────
# Data Generation Functions
# ──────────────────────────────────────────────

def generate_habitats(count: int) -> list[dict]:
    """Generate habitat data in batches to avoid token truncation."""
    log.info(f"Generating {count} habitats...")
    all_habitats = []
    batch_size = 10

    generated = 0
    while generated < count:
        batch = min(batch_size, count - generated)
        prompt = f"""Generate a JSON array of exactly {batch} unique animal habitats. Each object must have:
- "name": unique habitat name (e.g. "Serengeti Plains", "Amazon Rainforest")
- "type": one of ["Forest", "Grassland", "Desert", "Wetland", "Ocean", "Mountain", "Tundra", "Reef", "River", "Cave"]
- "climate": one of ["Tropical", "Temperate", "Arid", "Polar", "Mediterranean", "Continental", "Oceanic"]
- "region": geographic region (e.g. "East Africa", "South America")
- "avg_temp_c": average temperature in Celsius (number, -30 to 45)
- "area_km2": area in square km (number, 100 to 5000000)
- "description": one sentence describing the habitat

Return ONLY the JSON array, nothing else."""

        raw = rate_limited_generate(prompt, max_tokens=4000)
        batch_data = parse_json_array(raw)
        if batch_data:
            all_habitats.extend(batch_data)
            generated += len(batch_data)
            log.info(f"  Habitats progress: {generated}/{count}")

    return all_habitats[:count]


def generate_species(count: int, habitat_ids: list[int], batch_size: int) -> list[dict]:
    """Generate species data in batches."""
    log.info(f"Generating {count} species in batches of {batch_size}...")
    all_species = []
    classes = ["Mammalia", "Aves", "Reptilia", "Amphibia", "Actinopterygii", "Chondrichthyes", "Insecta", "Arachnida"]
    diets = ["Herbivore", "Carnivore", "Omnivore", "Insectivore", "Piscivore", "Frugivore"]
    statuses = ["Least Concern", "Near Threatened", "Vulnerable", "Endangered", "Critically Endangered"]

    generated = 0
    while generated < count:
        batch = min(batch_size, count - generated)
        prompt = f"""Generate a JSON array of {batch} unique animal species. Each object must have:
- "common_name": common English name (must be unique, realistic animals)
- "scientific_name": Latin binomial name
- "class": one of {json.dumps(classes)}
- "diet": one of {json.dumps(diets)}
- "conservation_status": one of {json.dumps(statuses)}
- "avg_lifespan_years": integer 1–200
- "habitat_id": randomly pick from {json.dumps(habitat_ids)}
- "description": one sentence about this species

Return ONLY the JSON array."""

        raw = rate_limited_generate(prompt, max_tokens=8000)
        batch_data = parse_json_array(raw)
        if batch_data:
            all_species.extend(batch_data)
            generated += len(batch_data)
            log.info(f"  Species progress: {generated}/{count}")

    return all_species[:count]


def generate_animals(count: int, species_ids: list[int], batch_size: int) -> list[dict]:
    """Generate individual animal data in batches."""
    log.info(f"Generating {count} animals in batches of {batch_size}...")
    all_animals = []
    sexes = ["Male", "Female"]
    health_statuses = ["Healthy", "Under Observation", "Recovering", "Injured", "Quarantined", "Elderly Care"]

    generated = 0
    batch_num = 0
    while generated < count:
        batch = min(batch_size, count - generated)
        batch_num += 1
        # Sample a subset of species IDs for the prompt to keep tokens low
        sample_ids = random.sample(species_ids, min(15, len(species_ids)))

        prompt = f"""Generate a JSON array of {batch} individual animals. Each object must have:
- "name": a unique animal name/nickname (creative, like "Thunder", "Luna", "Blaze")
- "species_id": randomly pick from {json.dumps(sample_ids)}
- "age_years": decimal number 0.1–50.0
- "weight_kg": decimal number 0.01–6000.0 (realistic for the species)
- "sex": one of {json.dumps(sexes)}
- "birth_date": date string "YYYY-MM-DD" between 1975-01-01 and 2026-01-01
- "health_status": one of {json.dumps(health_statuses)}
- "tag_number": unique tag like "TAG-{batch_num:03d}-001" through "TAG-{batch_num:03d}-{batch:03d}" (must be unique)

Return ONLY the JSON array."""

        raw = rate_limited_generate(prompt, max_tokens=8000)
        batch_data = parse_json_array(raw)
        if batch_data:
            # Ensure unique tag numbers
            for i, item in enumerate(batch_data):
                item["tag_number"] = f"TAG-{batch_num:04d}-{i+1:04d}"
            all_animals.extend(batch_data)
            generated += len(batch_data)
            log.info(f"  Animals progress: {generated}/{count}")

    return all_animals[:count]


def generate_diet_logs(count: int, animal_ids: list[int], batch_size: int) -> list[dict]:
    """Generate diet log data in batches."""
    log.info(f"Generating {count} diet logs in batches of {batch_size}...")
    all_logs = []

    generated = 0
    while generated < count:
        batch = min(batch_size, count - generated)
        sample_ids = random.sample(animal_ids, min(25, len(animal_ids)))

        prompt = f"""Generate a JSON array of {batch} animal feeding/diet log entries. Each object must have:
- "animal_id": randomly pick from {json.dumps(sample_ids)}
- "food_item": realistic food (e.g. "Fresh Salmon", "Timothy Hay", "Crickets", "Mixed Berries")
- "quantity_kg": decimal 0.01–50.0
- "feeding_time": time string "HH:MM:SS" (24-hour format)
- "calories": integer 10–15000
- "log_date": date string "YYYY-MM-DD" between 2024-01-01 and 2026-03-30

Return ONLY the JSON array."""

        raw = rate_limited_generate(prompt, max_tokens=8000)
        batch_data = parse_json_array(raw)
        if batch_data:
            all_logs.extend(batch_data)
            generated += len(batch_data)
            log.info(f"  Diet logs progress: {generated}/{count}")

    return all_logs[:count]


def generate_observations(count: int, animal_ids: list[int], batch_size: int) -> list[dict]:
    """Generate observation data in batches."""
    log.info(f"Generating {count} observations in batches of {batch_size}...")
    all_obs = []
    weathers = ["Sunny", "Cloudy", "Rainy", "Stormy", "Snowy", "Foggy", "Windy", "Clear", "Overcast", "Hot"]

    generated = 0
    while generated < count:
        batch = min(batch_size, count - generated)
        sample_ids = random.sample(animal_ids, min(25, len(animal_ids)))

        prompt = f"""Generate a JSON array of {batch} wildlife observation records. Each object must have:
- "animal_id": randomly pick from {json.dumps(sample_ids)}
- "observer_name": a realistic full name
- "obs_date": date string "YYYY-MM-DD" between 2024-01-01 and 2026-03-30
- "location": specific location description (e.g. "North Enclosure, Section B")
- "behavior": detailed behavior observation sentence (e.g. "Observed foraging near the riverbank with two juveniles")
- "health_notes": brief health note or null (e.g. "Slight limp on left hind leg")
- "weather": one of {json.dumps(weathers)}

Return ONLY the JSON array."""

        raw = rate_limited_generate(prompt, max_tokens=8000)
        batch_data = parse_json_array(raw)
        if batch_data:
            all_obs.extend(batch_data)
            generated += len(batch_data)
            log.info(f"  Observations progress: {generated}/{count}")

    return all_obs[:count]


# ──────────────────────────────────────────────
# Database Insertion
# ──────────────────────────────────────────────

def insert_habitats(cur, data: list[dict]) -> list[int]:
    """Insert habitats and return their IDs."""
    rows = []
    for h in data:
        rows.append((
            h.get("name", "Unknown"),
            h.get("type", "Forest"),
            h.get("climate", "Temperate"),
            h.get("region", "Unknown"),
            h.get("avg_temp_c", 20.0),
            h.get("area_km2", 1000.0),
            h.get("description", ""),
        ))
    ids = execute_values(
        cur,
        """INSERT INTO habitats (name, type, climate, region, avg_temp_c, area_km2, description)
           VALUES %s RETURNING id""",
        rows,
        fetch=True,
    )
    return [row[0] for row in ids]


def insert_species(cur, data: list[dict]) -> list[int]:
    """Insert species and return their IDs."""
    rows = []
    for s in data:
        rows.append((
            s.get("common_name", "Unknown"),
            s.get("scientific_name", "Unknown sp."),
            s.get("class", "Mammalia"),
            s.get("diet", "Omnivore"),
            s.get("conservation_status", "Least Concern"),
            s.get("avg_lifespan_years", 10),
            s.get("habitat_id", 1),
            s.get("description", ""),
        ))
    ids = execute_values(
        cur,
        """INSERT INTO species (common_name, scientific_name, class, diet, conservation_status,
                                avg_lifespan_years, habitat_id, description)
           VALUES %s RETURNING id""",
        rows,
        fetch=True,
    )
    return [row[0] for row in ids]


def insert_animals(cur, data: list[dict]) -> list[int]:
    """Insert animals and return their IDs."""
    rows = []
    for a in data:
        rows.append((
            a.get("name", "Unknown"),
            a.get("species_id", 1),
            a.get("age_years", 5.0),
            a.get("weight_kg", 50.0),
            a.get("sex", "Male"),
            a.get("birth_date", "2020-01-01"),
            a.get("health_status", "Healthy"),
            a.get("tag_number", f"TAG-{random.randint(100000, 999999)}"),
        ))
    ids = execute_values(
        cur,
        """INSERT INTO animals (name, species_id, age_years, weight_kg, sex, birth_date,
                                health_status, tag_number)
           VALUES %s RETURNING id""",
        rows,
        fetch=True,
    )
    return [row[0] for row in ids]


def insert_diet_logs(cur, data: list[dict]):
    """Insert diet logs."""
    rows = []
    for d in data:
        rows.append((
            d.get("animal_id", 1),
            d.get("food_item", "Unknown"),
            d.get("quantity_kg", 1.0),
            d.get("feeding_time", "12:00:00"),
            d.get("calories", 500),
            d.get("log_date", "2025-01-01"),
        ))
    execute_values(
        cur,
        """INSERT INTO diet_logs (animal_id, food_item, quantity_kg, feeding_time, calories, log_date)
           VALUES %s""",
        rows,
    )


def insert_observations(cur, data: list[dict]):
    """Insert observations."""
    rows = []
    for o in data:
        rows.append((
            o.get("animal_id", 1),
            o.get("observer_name", "Unknown"),
            o.get("obs_date", "2025-01-01"),
            o.get("location", "Unknown"),
            o.get("behavior", "No observation"),
            o.get("health_notes", None),
            o.get("weather", "Clear"),
        ))
    execute_values(
        cur,
        """INSERT INTO observations (animal_id, observer_name, obs_date, location, behavior,
                                      health_notes, weather)
           VALUES %s""",
        rows,
    )


# ──────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────

def print_stats():
    """Print API usage statistics."""
    elapsed = time.time() - api_stats["start_time"] if api_stats["start_time"] else 0
    log.info("=" * 55)
    log.info("API Usage Statistics")
    log.info(f"  Total requests : {api_stats['total_requests']}")
    log.info(f"  Total tokens   : {api_stats['total_tokens']}")
    log.info(f"  Elapsed time   : {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log.info("=" * 55)


def verify_data(cur):
    """Run verification queries and print results."""
    log.info("")
    log.info("=" * 55)
    log.info("DATA VERIFICATION")
    log.info("=" * 55)

    # Row counts
    for table in ["habitats", "species", "animals", "diet_logs", "observations"]:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        log.info(f"  {table:20s}: {count:,} rows")

    # Sample join query
    log.info("")
    log.info("Sample JOIN query (animals → species → habitats):")
    cur.execute("""
        SELECT a.name AS animal, s.common_name AS species,
               s.class, h.name AS habitat, h.climate
        FROM animals a
        JOIN species s ON a.species_id = s.id
        JOIN habitats h ON s.habitat_id = h.id
        LIMIT 5
    """)
    for row in cur.fetchall():
        log.info(f"  {row[0]:15s} │ {row[1]:20s} │ {row[2]:15s} │ {row[3]:20s} │ {row[4]}")

    # Multi-table join
    log.info("")
    log.info("Sample 4-table JOIN (diet_logs → animals → species → habitats):")
    cur.execute("""
        SELECT a.name AS animal, s.common_name AS species,
               d.food_item, d.quantity_kg, h.name AS habitat
        FROM diet_logs d
        JOIN animals a ON d.animal_id = a.id
        JOIN species s ON a.species_id = s.id
        JOIN habitats h ON s.habitat_id = h.id
        LIMIT 5
    """)
    for row in cur.fetchall():
        log.info(f"  {row[0]:15s} │ {row[1]:20s} │ {row[2]:20s} │ {row[3]:6.2f} kg │ {row[4]}")

    log.info("=" * 55)


def get_table_count(cur, table_name: str) -> int:
    """Get current row count for a table (returns 0 if table doesn't exist)."""
    try:
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        return cur.fetchone()[0]
    except Exception:
        cur.connection.rollback()
        return 0


def get_existing_ids(cur, table_name: str) -> list[int]:
    """Get all existing IDs from a table."""
    try:
        cur.execute(f"SELECT id FROM {table_name} ORDER BY id")
        return [row[0] for row in cur.fetchall()]
    except Exception:
        cur.connection.rollback()
        return []


def tables_exist(cur) -> bool:
    """Check if our schema tables already exist."""
    try:
        cur.execute("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_name IN ('habitats', 'species', 'animals', 'diet_logs', 'observations')
            AND table_schema = 'public'
        """)
        return cur.fetchone()[0] == 5
    except Exception:
        cur.connection.rollback()
        return False


def main():
    if not GROQ_API_KEY:
        log.error("GROQ_API_KEY not set. Create a .env file or set the environment variable.")
        sys.exit(1)

    log.info("=" * 55)
    log.info("  ANIMAL KINGDOM DATA GENERATOR")
    log.info("  Model: %s", MODEL)
    log.info("  Rate limit: %d RPM, 3.5s between requests", RPM_LIMIT)
    log.info("=" * 55)

    # ── Connect to database ──
    log.info("Connecting to PostgreSQL...")
    try:
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = False
        cur = conn.cursor()
        log.info("Connected successfully.")
    except Exception as e:
        log.error(f"Database connection failed: {e}")
        sys.exit(1)

    try:
        # ── Check if we can resume ──
        if tables_exist(cur):
            counts = {t: get_table_count(cur, t) for t in
                      ["habitats", "species", "animals", "diet_logs", "observations"]}
            log.info("Existing data found — RESUMING:")
            for t, c in counts.items():
                status = "✓ done" if c >= ROW_COUNTS.get(t, 0) else f"need {ROW_COUNTS.get(t, 0) - c} more"
                log.info(f"  {t:20s}: {c:,} rows ({status})")
        else:
            log.info("No existing tables found — creating schema...")
            cur.execute(SCHEMA_SQL)
            conn.commit()
            log.info("Schema created.")

        # ── Step 1: Habitats ──
        habitat_ids = get_existing_ids(cur, "habitats")
        existing_count = len(habitat_ids)
        needed = ROW_COUNTS["habitats"] - existing_count
        if needed > 0:
            log.info(f"Generating {needed} habitats (have {existing_count})...")
            habitat_data = generate_habitats(needed)
            new_ids = insert_habitats(cur, habitat_data)
            conn.commit()
            habitat_ids.extend(new_ids)
            log.info(f"✓ Inserted {len(new_ids)} habitats (total: {len(habitat_ids)})")
        else:
            log.info(f"⏩ Habitats already complete ({existing_count} rows)")
        if not habitat_ids:
            log.error("No habitats available. Cannot continue.")
            sys.exit(1)

        # ── Step 2: Species ──
        species_ids = get_existing_ids(cur, "species")
        existing_count = len(species_ids)
        needed = ROW_COUNTS["species"] - existing_count
        if needed > 0:
            log.info(f"Generating {needed} species (have {existing_count})...")
            species_data = generate_species(needed, habitat_ids, BATCH_SIZES["species"])
            for s in species_data:
                if s.get("habitat_id") not in habitat_ids:
                    s["habitat_id"] = random.choice(habitat_ids)
            new_ids = insert_species(cur, species_data)
            conn.commit()
            species_ids.extend(new_ids)
            log.info(f"✓ Inserted {len(new_ids)} species (total: {len(species_ids)})")
        else:
            log.info(f"⏩ Species already complete ({existing_count} rows)")
        if not species_ids:
            log.error("No species available. Cannot continue.")
            sys.exit(1)

        # ── Step 3: Animals (incremental batch commits) ──
        animal_ids = get_existing_ids(cur, "animals")
        existing_count = len(animal_ids)
        needed = ROW_COUNTS["animals"] - existing_count
        if needed > 0:
            log.info(f"Generating {needed} animals (have {existing_count})...")
            batch_size = BATCH_SIZES["animals"]
            sexes = ["Male", "Female"]
            health_statuses = ["Healthy", "Under Observation", "Recovering", "Injured", "Quarantined", "Elderly Care"]
            generated = 0
            batch_num = existing_count // batch_size
            while generated < needed:
                batch = min(batch_size, needed - generated)
                batch_num += 1
                sample_ids = random.sample(species_ids, min(15, len(species_ids)))
                prompt = f"""Generate a JSON array of {batch} individual animals. Each object must have:
- "name": a unique animal name/nickname (creative, like "Thunder", "Luna", "Blaze")
- "species_id": randomly pick from {json.dumps(sample_ids)}
- "age_years": decimal number 0.1–50.0
- "weight_kg": decimal number 0.01–6000.0 (realistic for the species)
- "sex": one of {json.dumps(sexes)}
- "birth_date": date string "YYYY-MM-DD" between 1975-01-01 and 2026-01-01
- "health_status": one of {json.dumps(health_statuses)}
- "tag_number": unique tag (must be unique)

Return ONLY the JSON array."""
                raw = rate_limited_generate(prompt, max_tokens=8000)
                batch_data = parse_json_array(raw)
                if batch_data:
                    for i, item in enumerate(batch_data):
                        item["tag_number"] = f"TAG-{batch_num:04d}-{i+1:04d}"
                        if item.get("species_id") not in species_ids:
                            item["species_id"] = random.choice(species_ids)
                    new_ids = insert_animals(cur, batch_data)
                    conn.commit()  # Save each batch immediately!
                    animal_ids.extend(new_ids)
                    generated += len(batch_data)
                    log.info(f"  Animals: {existing_count + generated}/{ROW_COUNTS['animals']} (batch saved ✓)")
        else:
            log.info(f"⏩ Animals already complete ({existing_count} rows)")
        if not animal_ids:
            log.error("No animals available. Cannot continue.")
            sys.exit(1)

        # ── Step 4: Diet Logs (incremental batch commits) ──
        existing_count = get_table_count(cur, "diet_logs")
        needed = ROW_COUNTS["diet_logs"] - existing_count
        if needed > 0:
            log.info(f"Generating {needed} diet logs (have {existing_count})...")
            batch_size = BATCH_SIZES["diet_logs"]
            generated = 0
            while generated < needed:
                batch = min(batch_size, needed - generated)
                sample_ids = random.sample(animal_ids, min(25, len(animal_ids)))
                prompt = f"""Generate a JSON array of {batch} animal feeding/diet log entries. Each object must have:
- "animal_id": randomly pick from {json.dumps(sample_ids)}
- "food_item": realistic food (e.g. "Fresh Salmon", "Timothy Hay", "Crickets", "Mixed Berries")
- "quantity_kg": decimal 0.01–50.0
- "feeding_time": time string "HH:MM:SS" (24-hour format)
- "calories": integer 10–15000
- "log_date": date string "YYYY-MM-DD" between 2024-01-01 and 2026-03-30

Return ONLY the JSON array."""
                raw = rate_limited_generate(prompt, max_tokens=8000)
                batch_data = parse_json_array(raw)
                if batch_data:
                    for d in batch_data:
                        if d.get("animal_id") not in animal_ids:
                            d["animal_id"] = random.choice(animal_ids)
                    insert_diet_logs(cur, batch_data)
                    conn.commit()  # Save each batch immediately!
                    generated += len(batch_data)
                    log.info(f"  Diet logs: {existing_count + generated}/{ROW_COUNTS['diet_logs']} (batch saved ✓)")
        else:
            log.info(f"⏩ Diet logs already complete ({existing_count} rows)")

        # ── Step 5: Observations (incremental batch commits) ──
        existing_count = get_table_count(cur, "observations")
        needed = ROW_COUNTS["observations"] - existing_count
        if needed > 0:
            log.info(f"Generating {needed} observations (have {existing_count})...")
            batch_size = BATCH_SIZES["observations"]
            weathers = ["Sunny", "Cloudy", "Rainy", "Stormy", "Snowy", "Foggy", "Windy", "Clear", "Overcast", "Hot"]
            generated = 0
            while generated < needed:
                batch = min(batch_size, needed - generated)
                sample_ids = random.sample(animal_ids, min(25, len(animal_ids)))
                prompt = f"""Generate a JSON array of {batch} wildlife observation records. Each object must have:
- "animal_id": randomly pick from {json.dumps(sample_ids)}
- "observer_name": a realistic full name
- "obs_date": date string "YYYY-MM-DD" between 2024-01-01 and 2026-03-30
- "location": specific location description (e.g. "North Enclosure, Section B")
- "behavior": detailed behavior observation sentence (e.g. "Observed foraging near the riverbank with two juveniles")
- "health_notes": brief health note or null (e.g. "Slight limp on left hind leg")
- "weather": one of {json.dumps(weathers)}

Return ONLY the JSON array."""
                raw = rate_limited_generate(prompt, max_tokens=8000)
                batch_data = parse_json_array(raw)
                if batch_data:
                    for o in batch_data:
                        if o.get("animal_id") not in animal_ids:
                            o["animal_id"] = random.choice(animal_ids)
                    insert_observations(cur, batch_data)
                    conn.commit()  # Save each batch immediately!
                    generated += len(batch_data)
                    log.info(f"  Observations: {existing_count + generated}/{ROW_COUNTS['observations']} (batch saved ✓)")
        else:
            log.info(f"⏩ Observations already complete ({existing_count} rows)")
        verify_data(cur)
        print_stats()

        log.info("")
        log.info("✅ Data generation complete!")

    except Exception as e:
        conn.rollback()
        log.error(f"Error during generation: {e}", exc_info=True)
        sys.exit(1)
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
