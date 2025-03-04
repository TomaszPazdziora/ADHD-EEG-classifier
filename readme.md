# ADHD EEG Classifier

## Project setup

Create a Virtual Environment

```bash
python3 -m venv venv
```

Activate venv

```bash
source venv/bin/activate # for Linux
# or 
venv\Scripts\activate # for Windows
```

Install libraries

```bash
pip install -r requirements.txt
```

## How to start

Currently 'model_training.py' is the main project file. Example runs:

```bash
python3 model_training.py --method forest
```

```bash
python3 model_training.py --method knn --opt
```

```bash
python3 model_training.py --method nn --hist
```

If get lost execute:
```bash
python3 model_training.py -h
```

