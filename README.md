# MAS-Project

Multi-Agent Systems (MAS)

## Team

- Nikethan
- Luca

## Create and activate Conda environment

```bash
conda create -n mas-project python=3.11
conda activate mas-project

pip install -r requirements.txt
```

## Start application

```bash
python main.py
```

## Metrics

Simulation metrics are stored as CSV files in the `metrics/` directory.

## Build executable application (optional)

```bash
pip install pyinstaller

pyinstaller --onefile --windowed main.py
```

## Build web application (optional)

```bash
pip install pygbag

python -m pygbag --build main.py
```

## Render and view report

```bash
quarto preview ./report/
```
