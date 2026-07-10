import pandas as pd

df = pd.read_csv("sensor_anomaly_results.csv")

df["anomaly_score"] = 0

df.loc[
    df["anomaly_flag"] == -1,
    "anomaly_score"
] = 100

df.loc[
    df["anomaly_flag"] == 1,
    "anomaly_score"
] = 0

df.to_csv(
    "../sensor_anomaly_results.csv",
    index=False
)

print("Scores Added")