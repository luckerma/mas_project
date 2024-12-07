# MAS-Project

Multi-Agent Systems (MAS)

The simulation is available at:\
https://luckerma.github.io/mas_project/

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

## Preview / Render Report (Quarto)

```bash
quarto preview ./report/

quarto render ./report/ --to pdf
```
