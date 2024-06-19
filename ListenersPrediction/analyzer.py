import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

from typing import Tuple, Any
from config import *
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass


@dataclass
class SongFeatures:
    followers: int
    danceability: float
    loudness: float


class PopularityPredictor:
    def __init__(self, file: str):
        self.dataframe = self.load_and_clean_data(file)
        self.scaler = StandardScaler()
        self.features, self.target = self.prepare_data(self.dataframe)
        self.features_train, self.features_test, self.target_train, self.target_test = self.split_data(self.features,
                                                                                                       self.target)
        self.lr_model = LinearRegression()
        self.random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)

    def load_and_clean_data(self, file: str) -> pd.DataFrame:
        df = pd.read_json(file)
        df.drop_duplicates(inplace=True)
        filtered_df = df[
            (df['followers'] >= DOWN_FOLLOWERS) & (df['followers'] <= UPPER_FOLLOWERS) &
            (df['loudness'] >= DOWN_LOUDNESS) & (df['loudness'] <= UPPER_LOUDNESS) &
            (df['danceability'] >= DOWN_DANCEABILITY) &
            (df['popularity'] >= DOWN_POPULARITY) & (df['popularity'] <= UPPER_POPULARITY)
            ]

        filtered_df = self.remove_columns(filtered_df, ['key', 'track_id', 'artist', 'country'])
        return filtered_df

    def remove_columns(self, df: pd.DataFrame, columns_to_remove: list) -> pd.DataFrame:
        return df.drop(columns=columns_to_remove, errors='ignore')

    def plotter(self, filtered_df: pd.DataFrame, plot_type: str) -> None:
        if plot_type == "popularity_vs_followers":
            plt.figure(figsize=(10, 6))
            sb.scatterplot(x='followers', y='popularity', data=filtered_df)
            plt.xlabel('Followers')
            plt.ylabel('Popularity')
            plt.title('Popularity vs Followers')

        elif plot_type == "popularity_vs_danceability":
            plt.figure(figsize=(10, 6))
            sb.scatterplot(x='danceability', y='popularity', data=filtered_df)
            plt.xlabel('Danceability')
            plt.ylabel('Popularity')
            plt.title('Popularity vs Danceability')

        elif plot_type == "popularity_vs_loudness":
            plt.figure(figsize=(10, 6))
            sb.scatterplot(x='loudness', y='popularity', data=filtered_df)
            plt.xlabel('Loudness')
            plt.ylabel('Popularity')
            plt.title('Popularity vs Loudness')

        elif plot_type == "distribution_of_popularity":
            plt.figure(figsize=(10, 6))
            sb.histplot(filtered_df['popularity'], kde=True, bins=10, color='green')
            plt.xlabel('Popularity')
            plt.ylabel('Frequency')
            plt.title('Distribution of Popularity')

        elif plot_type == "popularity_by_genre":
            plt.figure(figsize=(10, 6))
            sb.boxplot(x='popularity', y='track_genre', data=filtered_df)
            plt.xlabel('Popularity')
            plt.ylabel('Track Genre')
            plt.title('Popularity Distribution by Genre')

        plt.tight_layout()
        plt.show()

    def calculate_correlation(self, filtered_df: pd.DataFrame) -> None:
        numeric_df = filtered_df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()

        # Correlation matrix
        plt.figure(figsize=(12, 8))
        sb.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        df = pd.get_dummies(df, columns=['track_genre'], drop_first=True)
        features = df[['followers', 'danceability', 'loudness']]
        target = df['popularity']
        features = self.scaler.fit_transform(features)
        return features, target

    def split_data(self, features: np.ndarray, target: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return train_test_split(features, target, test_size=0.2, random_state=42)

    def calculate_accuracy(self, target_true: np.ndarray, target_pred: np.ndarray, tolerance: int = 10) -> float:
        correct_predictions = np.abs(target_true - target_pred) <= tolerance
        accuracy = np.mean(correct_predictions) * 100
        return accuracy

    def train_model(self, model: Any, features_train: np.ndarray, target_train: np.ndarray) -> Any:
        model.fit(features_train, target_train)
        return model

    def evaluate_model(self, model: Any, features: np.ndarray, target: np.ndarray) -> None:
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        mse_scores = cross_val_score(model, features, target, scoring='neg_mean_squared_error', cv=kf)
        r2_scores = cross_val_score(model, features, target, scoring='r2', cv=kf)
        mse = -mse_scores.mean()
        r2 = r2_scores.mean()
        target_pred = model.predict(features)
        accuracy = self.calculate_accuracy(target, target_pred)

        print(f"{model.__class__.__name__} - MSE: {mse}, R2: {r2}, Accuracy: {accuracy}%")

        # Predicted vs Actual popularity
        plt.figure(figsize=(10, 6))
        plt.scatter(target, target_pred)
        plt.xlabel('Actual Popularity')
        plt.ylabel('Predicted Popularity')
        plt.title(f'{model.__class__.__name__}: Predicted vs Actual Popularity')
        plt.tight_layout()
        plt.show()

        # Error distribution
        errors = target - target_pred
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=20)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.tight_layout()
        plt.show()

    def predict_single_song_popularity(self, model: Any, song: SongFeatures) -> float:
        song_features = pd.DataFrame([vars(song)])
        song_features_scaled = self.scaler.transform(song_features)
        predicted_popularity = model.predict(song_features_scaled)
        return predicted_popularity[0]


if __name__ == "__main__":
    predictor = PopularityPredictor(READ_FILE)

    predictor.plotter(predictor.dataframe, "followers_vs_popularity")
    predictor.plotter(predictor.dataframe, "danceability_vs_popularity")
    predictor.plotter(predictor.dataframe, "loudness_vs_popularity")
    predictor.plotter(predictor.dataframe, "popularity_distribution")
    predictor.plotter(predictor.dataframe, "popularity_by_genre")

    predictor.calculate_correlation(predictor.dataframe)

    trained_lr_model = predictor.train_model(predictor.lr_model, predictor.features_train, predictor.target_train)
    trained_rf_model = predictor.train_model(predictor.random_forest_model, predictor.features_train,
                                             predictor.target_train)

    predictor.evaluate_model(trained_lr_model, predictor.features_test, predictor.target_test)
    predictor.evaluate_model(trained_rf_model, predictor.features_test, predictor.target_test)

    song = SongFeatures(followers=5000, danceability=0.8, loudness=-5.0)
    predicted_popularity = predictor.predict_single_song_popularity(trained_rf_model, song)
    print(f"Predicted Popularity: {predicted_popularity}")
