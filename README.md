# Customer Churn Prediction

A binary-classification project that estimates whether a bank customer is likely to churn. It trains a TensorFlow/Keras neural network on customer account data and exposes the saved model through a Streamlit interface.

**Live demo:** [churn-prediction-alok.streamlit.app](https://churn-prediction-alok.streamlit.app/)

> The app is a learning/demo project. A churn probability is a model estimate, not a business decision by itself.

## Features

- Predicts the probability that a customer will exit the bank.
- Applies the same fitted encoders and scaler used during training before inference.
- Provides an interactive Streamlit UI for entering customer attributes.
- Includes notebooks for model training and grid-search experimentation.
- Preserves TensorBoard event logs for inspecting training history.

## Project structure

| Path | Purpose |
| --- | --- |
| `churn_classification_st.py` | Streamlit application for interactive predictions. |
| `churn_classification.ipynb` | Data preparation, model training, and model export workflow. |
| `HyperParameterTuning.ipynb` | Grid-search experiment using SciKeras and `GridSearchCV`. |
| `Churn_Modelling.csv` | Training dataset (10,000 customer records). |
| `classification_model.h5` | Trained Keras model used by the app. |
| `*_classification.pkl` | Fitted gender encoder, geography encoder, and feature scaler used by the app. |
| `ClassificationLogs/` | TensorBoard training event logs. |
| `requirements.txt` | Python dependencies. |

## Dataset and preprocessing

The dataset contains the target column `Exited`, where `1` denotes a customer who churned and `0` denotes a customer who stayed. Before training, the workflow:

1. Removes identifiers and name fields: `RowNumber`, `CustomerId`, and `Surname`.
2. Label-encodes `Gender`.
3. One-hot encodes `Geography` (`France`, `Germany`, and `Spain`).
4. Splits the data into training and test sets with an 80/20 split (`random_state=42`).
5. Fits a `StandardScaler` on the training features and transforms both splits.

The Streamlit app loads the persisted encoders and scaler so its input columns and transformations match those used to train the saved model.

## Model

The training notebook defines a feed-forward neural network:

```text
12 input features -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(1, sigmoid)
```

It is trained with Adam (`learning_rate=0.01`), binary cross-entropy loss, accuracy monitoring, TensorBoard logging, and early stopping on validation loss (patience 20, restoring the best weights).

In the recorded notebook run, the highest validation accuracy shown in the training log is **86.55%**. Re-run the notebook to obtain results for your environment or changed data.

## Getting started

### Prerequisites

- Python 3.10 or newer is recommended.
- `pip`

### Install dependencies

From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Run the prediction app

```powershell
streamlit run churn_classification_st.py
```

Streamlit will print a local URL (typically `http://localhost:8501`). Enter a customer's details to see the churn probability and a threshold-based classification:

- probability greater than `0.50`: likely to churn
- probability at or below `0.50`: not likely to churn

## Retrain the model

Open and run [`churn_classification.ipynb`](churn_classification.ipynb) from top to bottom. This regenerates:

- `classification_model.h5`
- `label_encoder_gender_classification.pkl`
- `one_hot_encoder_geo_classification.pkl`
- `scaler_classification.pkl`

These artifacts must remain in the project root for `churn_classification_st.py` to work.

## Hyperparameter experiments

[`HyperParameterTuning.ipynb`](HyperParameterTuning.ipynb) searches combinations of:

- hidden-layer units: 16, 32, 64, or 128
- hidden-layer count: 1, 2, or 3
- epochs: 50 or 100
- 3-fold cross-validation

It creates separate `*_hyperparametertuning.pkl` preprocessing artifacts. These are experimental outputs; the Streamlit app uses the `*_classification.pkl` artifacts instead.

## TensorBoard

To inspect the saved event logs, run:

```powershell
tensorboard --logdir ClassificationLogs
```

Then visit the address TensorBoard prints in the terminal.

## Tech stack

- TensorFlow / Keras
- scikit-learn and SciKeras
- pandas and NumPy
- Streamlit
- TensorBoard
