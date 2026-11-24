"""Estimate movie revenue from budget, runtime, and release year.

The bundled TMDB CSV is read with Python's standard library so this module can
run even when native pandas, NumPy, or scikit-learn extensions are unavailable.
"""
import csv
import math
import random
from datetime import datetime
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "movies.csv" / "tmdb_5000_movies.csv"
_FEATURE_NAMES = ("budget", "runtime", "release_year")


def _load_rows():
    # Resolve the dataset relative to this file rather than the shell's cwd.
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Revenue dataset not found: {DATA_PATH}. "
            "Place tmdb_5000_movies.csv in the movies.csv folder."
        )

    rows = []
    with DATA_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                budget = float(row["budget"])
                runtime = float(row["runtime"])
                revenue = float(row["revenue"])
                release_year = datetime.strptime(
                    row["release_date"], "%Y-%m-%d"
                ).year
            except (KeyError, TypeError, ValueError):
                continue
            # Ignore missing or invalid observations so they cannot distort fit.
            if budget > 0 and runtime > 0 and revenue > 0:
                rows.append(([budget, runtime, float(release_year)], revenue))
    if len(rows) < 10:
        raise ValueError(
            f"Only {len(rows)} usable revenue rows found in {DATA_PATH}; "
            "at least 10 are needed to train the model."
        )
    return rows


def _solve_linear_system(matrix, vector):
    """Solve a small square system with pivoting to reduce numeric errors."""
    size = len(vector)
    augmented = [list(matrix[row]) + [vector[row]] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("The training data does not support a unique regression model.")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                current - factor * pivot_value
                for current, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][-1] for row in range(size)]


def _fit(features, targets):
    # Standardize inputs before solving the normal equations for stable scaling.
    means = [sum(row[column] for row in features) / len(features) for column in range(3)]
    scales = []
    for column, mean in enumerate(means):
        variance = sum((row[column] - mean) ** 2 for row in features) / len(features)
        scales.append(math.sqrt(variance) or 1.0)
    design = [
        [1.0] + [(row[column] - means[column]) / scales[column] for column in range(3)]
        for row in features
    ]
    width = len(design[0])
    gram = [[0.0] * width for _ in range(width)]
    target_products = [0.0] * width
    for row, target in zip(design, targets):
        for left in range(width):
            target_products[left] += row[left] * target
            for right in range(width):
                gram[left][right] += row[left] * row[right]
    coefficients = _solve_linear_system(gram, target_products)
    return means, scales, coefficients


def _predict(features, means, scales, coefficients):
    normalized = [
        (features[index] - means[index]) / scales[index]
        for index in range(3)
    ]
    return coefficients[0] + sum(
        coefficients[index + 1] * normalized[index] for index in range(3)
    )


def _train_model():
    rows = _load_rows()
    # Use a fixed shuffle so reported evaluation metrics are repeatable.
    random.Random(42).shuffle(rows)
    split_at = max(1, min(len(rows) - 1, int(len(rows) * 0.8)))
    train_rows, test_rows = rows[:split_at], rows[split_at:]
    train_features = [row[0] for row in train_rows]
    train_targets = [row[1] for row in train_rows]
    means, scales, coefficients = _fit(train_features, train_targets)

    actual = [row[1] for row in test_rows]
    predicted = [
        _predict(row[0], means, scales, coefficients) for row in test_rows
    ]
    errors = [observed - estimate for observed, estimate in zip(actual, predicted)]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    actual_mean = sum(actual) / len(actual)
    total_variation = sum((value - actual_mean) ** 2 for value in actual)
    r2 = 1.0 - sum(error * error for error in errors) / total_variation if total_variation else 0.0

    model = (means, scales, coefficients)
    metrics = {"mae": mae, "rmse": rmse, "r2": r2, "training_rows": len(train_rows), "test_rows": len(test_rows)}
    return model, metrics


# Train once when imported, then reuse the fitted model for each UI request.
_MODEL, model_metrics = _train_model()


def predict_revenue(budget, runtime, release_year):
    """Return an estimated nonnegative revenue in dollars."""
    values = (float(budget), float(runtime), float(release_year))
    if values[0] <= 0 or values[1] <= 0 or values[2] < 1800:
        raise ValueError("Budget and runtime must be positive, and release year must be 1800 or later.")
    # Linear models can extrapolate below zero; revenue itself cannot be negative.
    return max(0.0, _predict(values, *_MODEL))
