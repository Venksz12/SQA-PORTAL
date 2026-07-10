import os
import pandas as pd

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

CSV_PATH = os.path.join(
    BASE_DIR,
    "sensor_stream_data.csv"
)

df = pd.read_csv(CSV_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"])

CHECK_DATE = "2025-06-15"

current_date = pd.to_datetime(CHECK_DATE)

historical = df[
    df["timestamp"] < current_date
].sort_values("timestamp")

previous_10_days = historical.tail(10)

current_rows = df[
    df["timestamp"].dt.date ==
    current_date.date()
]

if len(current_rows) == 0:
    print("Date not found")
    exit()

sensor_cols = [
    "temperature",
    "vibration",
    "pressure",
    "rpm",
    "engine_load",
    "coolant_temp",
    "oil_pressure",
    "speed"
]

print("\n===== SENSOR TREND REPORT =====\n")

for col in sensor_cols:

    avg_val = previous_10_days[col].mean()

    current_val = current_rows[col].mean()

    deviation = (
        (current_val - avg_val)
        / avg_val
    ) * 100

    status = "NORMAL"

    if abs(deviation) > 30:
        status = "CRITICAL"

    elif abs(deviation) > 15:
        status = "WARNING"

    print(
        f"{col}: "
        f"avg={avg_val:.2f} "
        f"current={current_val:.2f} "
        f"deviation={deviation:.2f}% "
        f"status={status}"
    )