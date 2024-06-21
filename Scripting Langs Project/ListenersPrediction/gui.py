import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from analyzer import PopularityPredictor, SongFeatures
from config import READ_FILE, INFO_MESSAGE


class ListenersPredictionApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Listeners Prediction")
        self.root.geometry("700x500")
        self.root.resizable(False, False)

        self.bg_image = Image.open("background_listeners_prediciton.png")
        self.bg_image = self.bg_image.resize((700, 500))
        self.bg_img = ImageTk.PhotoImage(self.bg_image)
        self.bg_label = tk.Label(root, image=self.bg_img)
        self.bg_label.place(x=-2, y=0)

        self.predictor = PopularityPredictor(READ_FILE)
        self.trained_model = None

        # Font config
        self.font_label = ("Helvetica", 12)
        self.font_button = ("Helvetica", 10)

        # Model selection
        self.model_var = tk.StringVar()
        model_label = tk.Label(root, text="Model", font=self.font_label)
        model_label.place(x=25, y=100)
        self.model_combo = ttk.Combobox(root, textvariable=self.model_var,
                                        values=["Linear Regression", "Random Forest"], font=self.font_label)
        self.model_combo.place(x=150, y=98)

        # Train button
        train_button = tk.Button(root, text="Train", command=self.train_model, font=self.font_button)
        train_button.place(x=150, y=135)

        # Plotter selection
        self.plotter_var = tk.StringVar()
        self.plotter_combo = ttk.Combobox(root, textvariable=self.plotter_var,
                                          values=["Popularity vs Followers", "Popularity vs Danceability",
                                                  "Popularity vs Loudness", "Distribution of Popularity",
                                                  "Popularity by Genre"], font=self.font_label)
        self.plotter_combo.place(x=420, y=101)

        # Plot button
        plot_button = tk.Button(root, text="Plot", command=self.plot_data, font=self.font_button)
        plot_button.place(x=421, y=139)

        # Info button
        info_button = tk.Button(root, text="Info", command=self.show_info, font=self.font_button)
        info_button.place(x=652, y=452)

        # Trained model label
        self.trained_model_label = tk.Label(root, text="Trained model: None", font=self.font_label)
        self.trained_model_label.place(x=20, y=192)

        # Input fields
        self.loudness_var = tk.DoubleVar()
        self.followers_var = tk.IntVar()
        self.danceability_var = tk.DoubleVar()

        loudness_label = tk.Label(root, text="Loudness", font=self.font_label)
        loudness_label.place(x=20, y=260)
        self.loudness_entry = tk.Entry(root, textvariable=self.loudness_var, font=self.font_label)
        self.loudness_entry.place(x=147, y=257)

        followers_label = tk.Label(root, text="Followers", font=self.font_label)
        followers_label.place(x=20, y=295)
        self.followers_entry = tk.Entry(root, textvariable=self.followers_var, font=self.font_label)
        self.followers_entry.place(x=147, y=293)

        danceability_label = tk.Label(root, text="Danceability", font=self.font_label)
        danceability_label.place(x=20, y=325)
        self.danceability_entry = tk.Entry(root, textvariable=self.danceability_var, font=self.font_label)
        self.danceability_entry.place(x=147, y=329)

        # Predict button
        predict_button = tk.Button(root, text="Predict", command=self.predict_popularity, font=self.font_button)
        predict_button.place(x=148, y=367)

        # Predicted popularity label
        self.predicted_popularity_label = tk.Label(root, text="Predicted popularity: ", font=self.font_label)
        self.predicted_popularity_label.place(x=20, y=435)

    def train_model(self) -> None:
        model_name = self.model_var.get()
        if model_name == "Linear Regression":
            self.predictor.lr_model = self.predictor.train_model(self.predictor.lr_model, self.predictor.features_train,
                                                                 self.predictor.target_train)
            self.trained_model = self.predictor.lr_model
            self.trained_model_label.config(text="Trained model: Linear Regression")
        elif model_name == "Random Forest":
            self.predictor.random_forest_model = self.predictor.train_model(self.predictor.random_forest_model,
                                                                            self.predictor.features_train,
                                                                            self.predictor.target_train)
            self.trained_model = self.predictor.random_forest_model
            self.trained_model_label.config(text="Trained model: Random Forest")
        else:
            messagebox.showerror("Error", "Please select a model to train.")

    def plot_data(self) -> None:
        plot_type = self.plotter_var.get()
        if plot_type == "Popularity vs Followers":
            self.predictor.plotter(self.predictor.dataframe, "popularity_vs_followers")
        elif plot_type == "Popularity vs Danceability":
            self.predictor.plotter(self.predictor.dataframe, "popularity_vs_danceability")
        elif plot_type == "Popularity vs Loudness":
            self.predictor.plotter(self.predictor.dataframe, "popularity_vs_loudness")
        elif plot_type == "Distribution of Popularity":
            self.predictor.plotter(self.predictor.dataframe, "distribution_of_popularity")
        elif plot_type == "Popularity by Genre":
            self.predictor.plotter(self.predictor.dataframe, "popularity_by_genre")
        else:
            messagebox.showerror("Error", "Please select plot type.")

    @staticmethod
    def show_info() -> None:
        messagebox.showinfo("App info", INFO_MESSAGE)

    def predict_popularity(self) -> None:
        if not self.trained_model:
            messagebox.showerror("Error", "Please train a model first.")
            return

        song = SongFeatures(
            followers=self.followers_var.get(),
            danceability=self.danceability_var.get(),
            loudness=self.loudness_var.get()
        )
        predicted_popularity = self.predictor.predict_single_song_popularity(self.trained_model, song)
        self.predicted_popularity_label.config(text=f"Predicted popularity: {predicted_popularity:.2f}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ListenersPredictionApp(root)
    root.mainloop()
