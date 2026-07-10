import pandas as pd
import numpy as np

np.random.seed(42)

ROWS = 10000

supplier_ids = [f"SUP-{i:03d}" for i in range(1, 101)]

parts = [
    "V-Belt",
    "Wheel Bearing",
    "Brake Chamber",
    "Relay Valve",
    "Radiator Hose",
    "Wheel Speed Sensor",
    "Air Filter",
    "U-Joint",
    "Fuel/Water Separator"
]

failure_types = [
    "Wear",
    "Leak",
    "Breakage",
    "Sensor Failure",
    "Corrosion",
    "Misalignment"
]

data = pd.DataFrame({
    "claim_id":[f"CLM{i:05d}" for i in range(ROWS)],
    
    "dealer_id":np.random.randint(1000,2000,ROWS),

    "supplier_id":np.random.choice(
        supplier_ids,
        ROWS
    ),

    "part_name":np.random.choice(
        parts,
        ROWS
    ),

    "failure_type":np.random.choice(
        failure_types,
        ROWS
    ),

    "claim_amount":np.random.randint(
        1000,
        50000,
        ROWS
    ),

    "repair_cost":np.random.randint(
        500,
        30000,
        ROWS
    ),

    "claim_frequency":np.random.randint(
        1,
        20,
        ROWS
    ),

    "days_since_last_claim":np.random.randint(
        1,
        365,
        ROWS
    ),

    "failure_severity":np.random.randint(
        1,
        10,
        ROWS
    )
})

# Create target column

data["fraud_label"] = (
    (data["claim_frequency"] > 15)
    &
    (data["claim_amount"] > 30000)
).astype(int)

data.to_csv(
    "dealer_claims_10000.csv",
    index=False
)

print("Dealer claims dataset generated.")
print(data.head())