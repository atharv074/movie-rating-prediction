# Movie Rating Prediction with Python

## Project Overview

Movie Rating Prediction is a machine learning project that predicts IMDb movie ratings based on information such as genre, director, actors, release year, duration, and number of votes.

The project uses a Random Forest Regression model to analyse movie data, identify important features, and estimate movie ratings.

## Objectives

- Explore and understand movie datasets.
- Clean and preprocess real-world data.
- Apply machine learning techniques for regression.
- Evaluate model performance using standard metrics.
- Identify features that contribute to model predictions.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Dataset

The project uses an IMDb Indian Movies dataset containing information about movie titles, release years, genres, durations, ratings, votes, directors, and actors.

The dataset is stored locally in the `data/` directory and is not included in this repository.

## Project Workflow

1. Data loading and exploration
2. Data cleaning and preprocessing
3. Handling missing values
4. Feature encoding
5. Training a Random Forest Regression model
6. Model evaluation
7. Feature importance analysis
8. Movie rating prediction

## Model

**Random Forest Regressor**

The model uses numerical and categorical movie features to predict IMDb ratings.

## Model Performance

Results from the previous model experiment:

| Evaluation Metric              | Score |
| ------------------------------ | ----: |
| Mean Absolute Error (MAE)      | 0.820 |
| Root Mean Squared Error (RMSE) | 1.095 |
| R² Score                       | 0.355 |

These results are from the earlier model run and should be verified against the final Python script.

## Feature Importance

The most influential features in the earlier experiment included:

- Number of Votes
- Release Year
- Movie Duration
- Genre

Feature importance represents the model's relative reliance on these features, not proof of causation.

## Installation and Usage

Clone the repository:

```bash
git clone https://github.com/atharv074/movie-rating-prediction.git
```

Navigate to the project directory:

```bash
cd movie-rating-prediction
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Place the dataset at:

```text
data/Movie dataset.csv
```

Run the program:

```bash
python movie_rating_prediction.py
```

## Limitations

- Prediction accuracy depends on dataset quality and available features.
- The model's performance may vary for newer movies.
- Movie ratings are influenced by subjective factors that are difficult to quantify.

## Future Improvements

- Experiment with additional regression algorithms.
- Improve feature engineering.
- Perform hyperparameter tuning.
- Develop a web-based interface for movie rating predictions.

## Author

Computer Engineering Student | Aspiring Data Scientist
