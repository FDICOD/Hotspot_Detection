import numpy as np
import pandas as pd
from pathlib import Path

def generate_data(n_safe=1000, n_hotspot=100):

    # Generate base features with noise
    safe = np.random.normal(
        [50, 50, 0, 0], [8, 8, 12, 12], (n_safe, 4))

    hotspot = np.random.normal(
        [47, 44, 6, 8], [7, 9, 10, 10], (n_hotspot, 4))

    # Combine
    X = np.vstack([safe, hotspot])
    y = np.hstack([np.zeros(n_safe), np.ones(n_hotspot)])

    # Create dataframe
    df = pd.DataFrame(X, columns=["width", "spacing", "x_coord", "y_coord"])

    # 🔥 Feature engineering
    df["aspect_ratio"] = (df["width"] / (df["spacing"] + 1e-6)) + np.random.normal(0, 0.1, len(df))
    df["distance_from_origin"] = np.sqrt(df["x_coord"]**2 + df["y_coord"]**2) + np.random.normal(0, 5, len(df))
    df["area"] = df["width"] * df["spacing"] + np.random.normal(0, 50, len(df))
    df["distance_from_center"] = np.sqrt((df["x_coord"] - 10)**2 + (df["y_coord"] - 10)**2)

    # Labels
    df["label"] = y
    noise_idx = np.random.choice(len(y), int(0.1 * len(y)), replace=False)
    y[noise_idx] = 1 - y[noise_idx]

    return df


if __name__ == "__main__":
    # Generate dataset
    df = generate_data()

    # Save inside data/ folder (same directory as this script)
    save_path = Path(__file__).parent / "dataset.csv"
    df.to_csv(save_path, index=False)

    print("Dataset saved successfully at:", save_path)
    print("\nColumns:", df.columns.tolist())
    print("\nShape:", df.shape)