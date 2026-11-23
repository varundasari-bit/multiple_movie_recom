# Load the bundled TMDB files with Python's standard library. This avoids
# pandas and NumPy native extensions that are blocked in some Windows setups.
import ast
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "movies.csv"
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "in", "into", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "was", "with", "you", "your",
}


def _read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_list(value):
    """Parse TMDB's serialized object lists and return their name values."""
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
    if not isinstance(parsed, list):
        return []
    return parsed


def _names(value, *, director_only=False, limit=None):
    objects = _parse_list(value)
    if director_only:
        objects = [item for item in objects if item.get("job") == "Director"]
    result = [item.get("name", "").replace(" ", "") for item in objects]
    result = [name for name in result if name]
    return result[:limit] if limit else result


def _load_movies():
    movie_rows = _read_csv(DATA_DIR / "tmdb_5000_movies.csv")
    credit_rows = _read_csv(DATA_DIR / "tmdb_5000_credits.csv")
    # Join on movie_id because titles can differ slightly between the files.
    credits_by_id = {row["movie_id"]: row for row in credit_rows}
    results = []
    seen_titles = set()
    for movie in movie_rows:
        credit = credits_by_id.get(movie.get("id"))
        title = (movie.get("title") or "").strip()
        overview = movie.get("overview") or ""
        if not credit or not title or not overview or title.casefold() in seen_titles:
            continue
        seen_titles.add(title.casefold())
        tags = " ".join(
            overview.split()
            + _names(movie.get("genres"))
            + _names(movie.get("keywords"))
            + _names(credit.get("cast"), limit=3)
            + _names(credit.get("crew"), director_only=True)
        )
        # Keep word counts per movie so cosine similarity can be computed sparsely.
        terms = [word for word in _TOKEN_RE.findall(tags.casefold()) if word not in _STOP_WORDS]
        results.append({"title": title, "terms": Counter(terms)})
    return results


_movies = _load_movies()
# Keep the app-facing movie collection simple; Streamlit only needs the titles.
movies = [{"title": item["title"]} for item in _movies]


def recommend(movie, limit=5):
    """Return the most similar movie titles using sparse cosine similarity."""
    selected = next(
        (item for item in _movies if item["title"].casefold() == str(movie).casefold()),
        None,
    )
    if selected is None:
        return []

    source = selected["terms"]
    source_norm = math.sqrt(sum(count * count for count in source.values()))
    ranked = []
    for candidate in _movies:
        if candidate is selected:
            continue
        terms = candidate["terms"]
        candidate_norm = math.sqrt(sum(count * count for count in terms.values()))
        if not source_norm or not candidate_norm:
            score = 0.0
        else:
            dot = sum(count * terms.get(word, 0) for word, count in source.items())
            score = dot / (source_norm * candidate_norm)
        ranked.append((score, candidate["title"]))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [title for _, title in ranked[:max(0, limit)]]
