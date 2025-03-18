# andranik

andranik is a chess position detection project for getting the best move given just the picture of the position

## Usage

### Dependencies
- uv
- stockfish (local or global)

### Setup
Run 
```bash
uv sync
```
to syncronise the packages.  

To use, run (for global stockfish)
```bash
MODEL_PATH="ml/model" uv run main.py path/to/your/image.png
```
and run (for local stockfish)
```bash
MODEL_PATH="ml/model" STOCKFISH_PATH="path/to/stockfish" uv run main.py path/to/your/image.png
```
