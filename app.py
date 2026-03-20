"""
Streamlit PM2.5 Prediction App
================================
Loads the trained LSTM model from pm25_model.pkl and provides two modes:
  1. Enter a specific time (HH:MM) → get a predicted PM2.5 value
  2. Upload a CSV/Excel file → get PM2.5 predictions for the entire period
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st

# ─── LSTM model definition (must match the notebook) ───────────────────────────

class AQI_LSTM(nn.Module):
    """LSTM model for PM2.5 prediction."""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


# ─── Helper functions ──────────────────────────────────────────────────────────

def get_pm25_category(pm25):
    """Map PM2.5 concentration to Indian AQI category (PM2.5 breakpoints)."""
    if pm25 <= 30:
        return "Good"
    elif pm25 <= 60:
        return "Satisfactory"
    elif pm25 <= 90:
        return "Moderate"
    elif pm25 <= 120:
        return "Poor"
    elif pm25 <= 250:
        return "Very Poor"
    else:
        return "Severe"


def predict_for_time(time_str, model, train_data, features, seq_len,
                     feat_mean, feat_std, tgt_mean, tgt_std):
    """Predict PM2.5 for a given time string (HH:MM)."""
    h, m = map(int, time_str.strip().split(":"))
    total_min = h * 60 + m

    # Cyclical encoding
    t_sin = np.sin(2 * np.pi * total_min / 1440)
    t_cos = np.cos(2 * np.pi * total_min / 1440)

    # Find nearby training data points (±60 min)
    train_copy = train_data.copy()
    train_copy["total_min"] = train_copy["hour"] * 60 + train_copy["minute"]
    time_diff = np.abs(train_copy["total_min"] - total_min)
    time_diff = np.minimum(time_diff, 1440 - time_diff)

    nearby = train_copy[time_diff <= 60]
    if len(nearby) == 0:
        nearby = train_copy[time_diff <= 120]

    # Build input sequence
    if len(nearby) >= seq_len:
        closest = nearby.iloc[time_diff[time_diff <= 60].argsort().values[:seq_len]]
        seq = closest[features].values.astype(np.float32)
    else:
        avg = nearby[features].mean().values.astype(np.float32)
        avg[0], avg[1] = t_sin, t_cos
        seq = np.tile(avg, (seq_len, 1))

    # Normalise & predict
    seq_norm = (seq - feat_mean) / feat_std
    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(seq_norm).unsqueeze(0)).item()

    pm25 = max(0, pred * tgt_std + tgt_mean)
    return pm25


# ─── Load model assets ─────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    """Load pickle and rebuild the LSTM model."""
    pkl_path = os.path.join(os.path.dirname(__file__), "pm25_model.pkl")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    model = AQI_LSTM(
        input_size=data["input_size"],
        hidden_size=data["hidden_size"],
        num_layers=data["num_layers"],
        dropout=data["dropout"],
    )
    model.load_state_dict(data["model_state_dict"])
    model.eval()

    return (
        model,
        pd.DataFrame(data["train_data_dict"]),
        data["features"],
        data["seq_len"],
        data["feat_mean"],
        data["feat_std"],
        data["tgt_mean"],
        data["tgt_std"],
    )


# ─── Streamlit UI ──────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="PM2.5 Predictor", page_icon="🌫️", layout="centered")
    st.title("🌫️ PM2.5 Prediction System")
    st.caption("Bengaluru – Jayanagar 5th Block KSPCB  •  LSTM model")

    model, train_data, features, seq_len, feat_mean, feat_std, tgt_mean, tgt_std = load_model()

    tab1, tab2 = st.tabs(["⏰ Predict by Time", "📄 Upload File"])

    # ── Tab 1: single time ──
    with tab1:
        st.subheader("Enter a time of day")
        time_input = st.text_input("Time (HH:MM, 24-hour format)", value="14:30")

        if st.button("Predict PM2.5", key="btn_single"):
            try:
                parts = time_input.strip().split(":")
                h_val, m_val = int(parts[0]), int(parts[1])
                if not (0 <= h_val <= 23 and 0 <= m_val <= 59):
                    st.error("Invalid time. Hours 0-23, Minutes 0-59.")
                else:
                    pm25 = predict_for_time(
                        time_input, model, train_data, features,
                        seq_len, feat_mean, feat_std, tgt_mean, tgt_std,
                    )
                    category = get_pm25_category(pm25)

                    col1, col2 = st.columns(2)
                    col1.metric("Predicted PM2.5 (µg/m³)", f"{pm25:.1f}")
                    col2.metric("Category", category)
            except Exception as exc:
                st.error(f"Error: {exc}")

    # ── Tab 2: file upload ──
    with tab2:
        st.subheader("Upload a file")
        st.markdown(
            "Upload a **CSV or Excel** file.  \n"
            "Supported formats:  \n"
            "- Raw sensor data (same format as the training CSVs, with `datetimeLocal` and `parameter` columns)  \n"
            "- A simple file with a **Time** column containing HH:MM values"
        )

        uploaded = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])

        if uploaded is not None:
            try:
                if uploaded.name.endswith(".csv"):
                    df = pd.read_csv(uploaded)
                else:
                    df = pd.read_excel(uploaded)

                # ── Format A: raw sensor data (same as training CSVs) ──
                if "datetimeLocal" in df.columns and "parameter" in df.columns:
                    # Preprocess like the notebook
                    df["datetime"] = pd.to_datetime(df["datetimeLocal"]).dt.tz_localize(None)
                    data = df.pivot_table(index="datetime", columns="parameter",
                                          values="value", aggfunc="mean")
                    data.sort_index(inplace=True)
                    data.dropna(inplace=True)
                    data = data.resample("15min").mean().interpolate(method="linear")

                    rename_map = {"pm10": "PM10", "pm25": "PM25",
                                  "relativehumidity": "RH", "temperature": "Temp",
                                  "wind_speed": "Wind"}
                    data.rename(columns=rename_map, inplace=True)

                    # Cyclical time
                    total_min = data.index.hour * 60 + data.index.minute
                    data["time_sin"] = np.sin(2 * np.pi * total_min / 1440)
                    data["time_cos"] = np.cos(2 * np.pi * total_min / 1440)
                    data["hour"] = data.index.hour
                    data["minute"] = data.index.minute

                    # Predict for every row
                    results = []
                    for idx, row in data.iterrows():
                        t_str = f"{int(row['hour']):02d}:{int(row['minute']):02d}"
                        pm25_pred = predict_for_time(
                            t_str, model, train_data, features,
                            seq_len, feat_mean, feat_std, tgt_mean, tgt_std,
                        )
                        results.append({
                            "Time": idx.strftime("%Y-%m-%d %H:%M"),
                            "Actual PM2.5": round(row["PM25"], 1),
                            "Predicted PM2.5": round(pm25_pred, 1),
                            "Error": round(abs(row["PM25"] - pm25_pred), 1),
                            "Category": get_pm25_category(pm25_pred),
                        })

                    result_df = pd.DataFrame(results)
                    st.success(f"Predicted PM2.5 for {len(result_df)} timestamps.")
                    st.dataframe(result_df, use_container_width=True)

                    csv_out = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download results as CSV", csv_out,
                                       file_name="pm25_predictions.csv", mime="text/csv")

                # ── Format B: simple file with a Time column ──
                elif "Time" in df.columns:
                    predictions = []
                    categories = []
                    for t in df["Time"]:
                        t_str = str(t).strip()
                        pm25 = predict_for_time(
                            t_str, model, train_data, features,
                            seq_len, feat_mean, feat_std, tgt_mean, tgt_std,
                        )
                        predictions.append(round(pm25, 1))
                        categories.append(get_pm25_category(pm25))

                    df["Predicted PM2.5"] = predictions
                    df["Category"] = categories
                    st.success(f"Predicted PM2.5 for {len(df)} time entries.")
                    st.dataframe(df, use_container_width=True)

                    csv_out = df.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download results as CSV", csv_out,
                                       file_name="pm25_predictions.csv", mime="text/csv")
                else:
                    st.error("Unsupported file format. Need either raw sensor columns "
                             "(`datetimeLocal`, `parameter`) or a `Time` column.")

            except Exception as exc:
                st.error(f"Error processing file: {exc}")


if __name__ == "__main__":
    main()
