# Iris CBR System

A Case-Based Reasoning (CBR) system using the Iris dataset, built with PyCBR and Flask. This project loads and processes the dataset, sets up a case base, applies a recovery strategy, and uses majority aggregation for classification.

## Features
- Loads the Iris dataset into a structured case base.
- Implements a recovery strategy using Quantile Linear Attributes.
- Uses Majority Aggregation for classification.
- Built with clean code principles for modularity and ease of use.

## Installation

```bash
# Clone the repository
git clone https://github.com/vallz0/iris-cbr-system.git
cd iris-cbr-system

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the application:

```bash
python main.py
```

Access the web interface at `http://127.0.0.1:5000/`.

## Dependencies
- Python 3.x
- PyCBR
- Pandas
- Scikit-learn
- Flask


