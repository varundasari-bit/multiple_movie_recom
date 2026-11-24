# Movie Recommendation System

A small Streamlit app that recommends movies with content similar to a selected movie. It compares each movie's overview, genres, keywords, top cast, and director, then displays the five closest matches.

## Project files

```text
movie recom/
|-- app.py                         # Streamlit user interface
|-- recommender.py                 # CSV loading and similarity calculation
|-- requirements.txt               # Python package requirements
|-- movies.csv/
    |-- tmdb_5000_movies.csv       # Movie titles, overviews, genres, keywords
    |-- tmdb_5000_credits.csv      # Cast and crew information
```

The two CSV files are expected in the included `movies.csv` folder. The recommender joins their rows by movie ID. If you move or replace the dataset, keep those filenames and folder layout.

## Requirements

- Windows with Python installed and available as `py` or `python`
- Packages listed in `requirements.txt` (currently Streamlit)

Recommendation processing uses Python's standard library. Streamlit supplies the web interface.

## Set up a virtual environment

Open PowerShell in the project folder. Since this folder name contains a space, quote the path when changing to it:

```powershell
cd "C:\Users\dasar\CodeFiles_VARUN\project\movie recom"
```

Create and activate a project-local virtual environment, then install the requirements:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell says script execution is disabled when activating the environment, skip activation and use the environment's Python directly in the run command below.

## Run the app

With the virtual environment activated:

```powershell
python -m streamlit run app.py
```

Or without activating it:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit prints a **Local URL**, usually `http://localhost:8501`. Open it in a browser, select a movie, and click **Recommend Movies**. Stop the server by pressing `Ctrl+C` in the PowerShell window.

## Troubleshooting

### Fatal error in launcher / old project path

If `streamlit run app.py` mentions a different or old project directory, Windows is finding a stale `streamlit.exe` launcher. Run Streamlit through the active Python instead:

```powershell
python -m streamlit run app.py
```

Make sure PowerShell is in this project folder and that the virtual environment is activated. You can confirm the Python being used with:

```powershell
python -c "import sys; print(sys.executable)"
```

### `No module named streamlit`

Activate the virtual environment and install dependencies again:

```powershell
python -m pip install -r requirements.txt
```

### Dataset file not found

Confirm both CSV files exist at these paths relative to the project folder:

```text
movies.csv\tmdb_5000_movies.csv
movies.csv\tmdb_5000_credits.csv
```

### Port 8501 is already in use

Stop the other Streamlit process with `Ctrl+C`, or start this app on another port:

```powershell
python -m streamlit run app.py --server.port 8502
```

## How recommendations are calculated

The recommender turns each movie's overview and selected metadata into word counts. It compares those word-count vectors with cosine similarity and returns the highest-scoring titles, excluding the selected movie itself. The computation is performed when a user requests recommendations.
