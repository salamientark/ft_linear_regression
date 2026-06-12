# FT_LINEAR_REGRESSION

Predict a car's price from its mileage using a simple linear regression
trained with the **Batch Gradient Descent** algorithm.

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/) — Python package & environment manager
- Python **3.14+** (uv installs it automatically if missing)

Python dependencies (installed by `uv sync`):

| Package    | Purpose                       |
|------------|-------------------------------|
| numpy      | Numerical computations        |
| pandas     | CSV loading / data handling   |
| matplotlib | Plotting data and regression  |

## Setup

From the project root:

```bash
uv sync
source .venv/bin/activate
```

`uv sync` creates the `.venv` virtual environment and installs all
dependencies from `pyproject.toml` / `uv.lock`. Sourcing the activate
script enables the environment in your current shell.

## Data

Training requires a `data.csv` file in the project root with two columns:

```csv
km,price
240000,3650
139800,3800
...
```

## Usage

Run all programs from the project root (they look for `data.csv` and
`.theta` in the current directory).

### 1. Train the model

```bash
python src/train.py
```

- Loads `data.csv`, normalizes the mileage (Z-score), then runs batch
  gradient descent (100 iterations, learning rate 0.01)
- Saves the resulting weights to the `.theta` file
- Prints the old vs. new theta values and plots the data points together
  with the old and new regression lines

### 2. Estimate a price

```bash
python src/estimate_price.py
```

- Prompts for a car mileage and prints the predicted price
- Uses the weights stored in `.theta` (if the file doesn't exist, thetas
  are initialized to 0 — train first for meaningful predictions)

### 3. Measure model precision (bonus)

```bash
python src/precision.py
```

Evaluates the trained model against `data.csv` and prints:
- **MSE** — Mean Square Error
- **RMSE** — Root Mean Square Error
- **MAE** — Mean Absolute Error
- **VAR** — Variance explained by the model (R²)

## Project structure

```
.
├── src/
│   ├── train.py           # Train the model (gradient descent)
│   ├── estimate_price.py  # Predict a price from a mileage
│   └── precision.py       # Model precision metrics
├── .theta                 # Saved model weights (theta0, theta1)
├── data.csv               # Training data (km, price) — user provided
└── pyproject.toml         # Project metadata & dependencies
```

## How it works

### What is gradient descent?

Gradient descent algorithms find the best parameters (weights/theta)
that minimize the loss function (error). Batch gradient descent goes
through **all** the data at each iteration: more stable and accurate,
but slower. That's fine here given the small dataset (24 rows).

### The model

We predict the price (output) from the km (feature), so the hypothesis is:

**price = θ0 + (θ1 × km)**

To allow a higher learning rate (faster convergence), the `km` feature is
normalized with Z-score scaling (standardization):

**normalized_x = (x - mean) / std**

The gradient update formulas (derived from the loss function) are:

**θ0 = θ0 - α × (1/m) × ∑(h(X) - price(X))**

**θ1 = θ1 - α × (1/m) × ∑((h(X) - price(X)) × km(X))**

with α the learning rate, m the dataset size, and h(X) the hypothesis.

After training, the thetas are converted back to the un-normalized scale
and saved — voilà, weights updated!
