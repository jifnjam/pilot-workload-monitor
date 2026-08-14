import customtkinter as ctk
import threading
import time
import json
import tkinter as tk
from pathlib import Path
import joblib
import mne
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image

# Set up paths to be used
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (13, 30),
    "gamma": (30, 40),
}

LABEL_NAMES = {1: "low", 2: "medium", 3: "high"}
LABEL_COLORS = {"low": "green", "medium": "yellow", "high": "red"}

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


def extract_band_features_for_epochs(epochs, feature_cols):
    """Applies same computation seen in 04_feature_extraction.ipynb in
    order to return a DataFrame of the band features of the epochs

    Parameters:
        epochs: epo-fif file of epochs
        feature_cols:

    Returns:
        DataFrame of feature_cols as columns"""

    ch_names = epochs.ch_names
    # psd computation
    spectrum = epochs.compute_psd(method="welch", fmin=1, fmax=40)
    psds, freqs = spectrum.get_data(return_freqs=True)
    n_epochs = psds.shape[0]
    rows = [dict() for i in range(n_epochs)]

    eps = 1e-8
    for band_name, (fmin, fmax) in BANDS.items():
        band_mask = (freqs >= fmin) & (freqs <= fmax)
        band_power = psds[:, :, band_mask].mean(axis=2)
        band_power_db = 10 * np.log10(band_power * 1e12 + eps)  # prevents overfitting
        for ch_idx, ch_name in enumerate(ch_names):
            col = f"{ch_name}_{band_name}"
            for epoch_idx in range(n_epochs):
                rows[epoch_idx][col] = band_power_db[epoch_idx, ch_idx]

    frontal = [c for c in ("EEG.AF3", "EEG.AF4") if c in ch_names]
    if len(frontal) == 2:
        theta_mask = (freqs >= BANDS["theta"][0]) & (freqs <= BANDS["theta"][1])
        beta_mask = (freqs >= BANDS["beta"][0]) & (freqs <= BANDS["beta"][1])
        frontal_idx = [ch_names.index(c) for c in frontal]
        frontal_theta = psds[:, frontal_idx][:, :, theta_mask].mean(axis=(1, 2))
        frontal_beta = psds[:, frontal_idx][:, :, beta_mask].mean(axis=(1, 2))
        ratio = frontal_theta / (frontal_beta + eps)
        ratio = np.nan_to_num(ratio, nan=0.0, posinf=100.0, neginf=0.0)
        ratio = np.clip(ratio, 0, 100)
        for epoch_idx in range(n_epochs):
            rows[epoch_idx]["frontal_theta_beta_ratio"] = ratio[epoch_idx]

    df = pd.DataFrame(rows)
    # select and sort the feature columns
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    return df[feature_cols]


class WorkloadMonitorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Pilot Cognitive Workload Monitor")
        self.geometry("1000x750")

        self.model = joblib.load(MODELS_DIR / "baseline_random_forest.joblib")
        self.feature_cols = list(self.model.feature_names_in_)

        # states
        self.epochs = None
        self.features = None
        self.current_idx = 0
        self.playing = False

        self.tabs = ctk.CTkTabview(self)
        self.tabs.pack(fill="both", expand=True, padx=10, pady=10)
        self.monitor_tab = self.tabs.add("Live Monitor")
        self.performance_tab = self.tabs.add("Model Performance")

        self.build_monitor_tab()
        self.build_performance_tab()

    def build_monitor_tab(self):
        top_frame = ctk.CTkFrame(self.monitor_tab)
        top_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(top_frame, text="Recording:").pack(side="left", padx=(0, 5))

        # dict for dropdown list
        epo_files = sorted(PROCESSED_DIR.glob("*-epo.fif"))
        self.recording_paths = {f.stem.replace("-epo", ""): f for f in epo_files}
        recording_names = list(self.recording_paths.keys())

        # recording management
        self.recording_dropdown = ctk.CTkOptionMenu(
            top_frame, values=recording_names, command=self.on_recording_selected
        )
        self.recording_dropdown.pack(side="left", padx=5)

        self.play_button = ctk.CTkButton(
            top_frame, text="Play", command=self.toggle_play, width=80
        )
        self.play_button.pack(side="left", padx=5)

        self.next_button = ctk.CTkButton(
            top_frame, text="Next Epoch", command=self.next_epoch, width=100
        )
        self.next_button.pack(side="left", padx=5)

        # progress label
        self.epoch_progress_label = ctk.CTkLabel(
            top_frame, text="Select a recording to begin"
        )
        self.epoch_progress_label.pack(side="right", padx=10)

        # band power bar chart visualization
        self.fig = Figure(figsize=(6, 3), dpi=100)
        self.ax = self.fig.add_subplot()
        # turn matplotlib figure into something better for tkinter usage
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.monitor_tab)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        gauge_frame = ctk.CTkFrame(self.monitor_tab)
        gauge_frame.pack(fill="x", padx=10, pady=10)

        self.gauge_canvas = tk.Canvas(
            gauge_frame, width=220, height=120, bg="black", highlightthickness=0
        )
        self.gauge_canvas.pack(side="left", padx=10, pady=10)

        self.prediction_label = ctk.CTkLabel(
            gauge_frame, text="", font=("", 20, "bold")
        )
        self.prediction_label.pack(side="left", padx=20)

        self.disclaim_label = ctk.CTkLabel(
            self.monitor_tab,
            text=(
                "Note: the model was only trained on task-phase epochs."
                "Predictions on baseline & recovery recordings are for exploration purposes only."
            ),
            text_color="gray55",
            font=("", 11),
        )
        self.disclaim_label.pack(pady=(0, 5))

    def on_recording_selected(self, name):
        self.playing = False
        self.play_button.configure(text="Play")
        self.epochs = mne.read_epochs(
            self.recording_paths[name], preload=True, verbose=False
        )
        self.features = extract_band_features_for_epochs(self.epochs, self.feature_cols)
        self.current_index = 0
        self.update_display()

    def next_epoch(self):
        if self.epochs is None:
            return
        # Modulus prevents a zero error
        self.current_index = (self.current_index + 1) % len(self.epochs)
        self.update_display()

    def toggle_play(self):
        if self.epochs is None:
            return
        self.playing = not self.playing
        self.play_button.configure(text="Pause" if self.playing else "Play")
        if self.playing:
            self.play_loop()

    def play_loop(self):
        if not self.playing or self.epochs is None:
            return
        self.next_epoch()
        # Instead of a live food the loop is based on 1 second per epoch
        # This is in the case of the Plan A from 03 notebook
        # Plan B needs threading if used
        self.after(1000, self.play_loop)

    def update_display(self):
        if self.epochs is None:
            return

        self.epoch_progress_label.configure(
            text=f"Epoch {self.current_index + 1} / {len(self.epochs)}"
        )

        # get an epoch
        row = self.features.iloc[self.current_index]
        band_avgs = {}
        for band in BANDS:
            band_cols = [c for c in self.feature_cols if c.endswith(f"_{band}")]
            band_avgs[band] = row[band_cols].mean()  # for plotting purposes
            # cannot plot all 70 bands together so an average is a better visual

        self.ax.clear()  # for resetting
        self.ax.bar(list(band_avgs.keys()), list(band_avgs.values()), color="blue")
        self.ax.set_ylabel("Power (dB)")
        self.ax.set_title("Band power | current epoch (averaged across channels)")
        self.canvas.draw()

        X_row = row.to_frame().T
        pred = self.model.predict(X_row)[0]

        label = LABEL_NAMES[pred]
        self.prediction_label.configure(
            text=f"Predicted workload: {label.upper()}", text_color=LABEL_COLORS[label]
        )
        self.draw_gauge(label)

    def draw_gauge(self, active_label):
        self.gauge_canvas.delete("all")
        positions = {"low": 0, "medium": 1, "high": 2}
        # Make boxes to represent workload levels
        for name, pos in positions.items():
            if name == active_label:
                fill = LABEL_COLORS[name]
            else:
                fill = "gray"
            x0 = 10 + pos * 70
            self.gauge_canvas.create_rectangle(
                x0, 35, x0 + 60, 80, fill=fill, outline=""
            )
            self.gauge_canvas.create_text(x0 + 30, 100, text=name, fill="white")

    # model performance tab
    def build_performance_tab(self):
        metrics_path = DOCS_DIR / "metrics.json"

        if not metrics_path.exists():
            ctk.CTkLabel(
                self.performance_tab,
                text="No docs/metrics.json was found | run 05_model_training.ipynb first.",
            ).pack(pady=20)
            return

        with open(metrics_path) as f:
            metrics = json.load(f)

        summary = f"""Cross-validated accuracy: {metrics['cv_accuracy']:.1%}\n
            Cross-validated macro F1 score: {metrics['cv_macro_f1']:.1%}\n
            Samples: {metrics['n_samples']}\n
            Features: {metrics['n_features']}\n
            Subjects: {metrics['n_subjects']}\n
            CV folds: {metrics['cv_folds']}\n"""
        ctk.CTkLabel(
            self.performance_tab, text=summary, justify="left", font=("", 14)
        ).pack(pady=15, anchor="w", padx=15)

        images_frame = ctk.CTkFrame(self.performance_tab)
        images_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Keeping the references on self
        # Clears pictures if not available
        cm_path = DOCS_DIR / "screenshots" / "confusion_matrix.png"
        fi_path = DOCS_DIR / "screenshots" / "feature_importance.png"
        if cm_path.exists():
            self.cm_image = ctk.CTkImage(Image.open(cm_path), size=(400, 350))
            ctk.CTkLabel(images_frame, image=self.cm_image, text="").pack(
                side="left", padx=10, pady=10
            )
        else:
            ctk.CTkLabel(images_frame, txt="confusion_matrix.png not found").pack(
                side="left", padx=10
            )

        if fi_path.exists():
            self._fi_image = ctk.CTkImage(Image.open(fi_path), size=(450, 300))
            ctk.CTkLabel(images_frame, image=self._fi_image, text="").pack(
                side="left", padx=10, pady=10
            )
        else:
            ctk.CTkLabel(images_frame, text="feature_importance.png not found").pack(
                side="left", padx=10
            )


if __name__ == "__main__":
    app = WorkloadMonitorApp()
    app.mainloop()
