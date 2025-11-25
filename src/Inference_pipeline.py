import pandas as pd
import joblib
import os



# Load Artifacts (Encoders, Scaler, Model)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

label_encoders = joblib.load(
    os.path.join(BASE_DIR, "../Artifacts/encoders/train_label_encoders.pkl")
)

price_le = joblib.load(
    os.path.join(BASE_DIR, "../Artifacts/encoders/price_range_label_encoder.pkl")
)

ohe = joblib.load(
    os.path.join(BASE_DIR, "../Artifacts/encoders/onehot_encoder.pkl")
)

scaler = joblib.load(
    os.path.join(BASE_DIR, "../Artifacts/encoders/scaler.pkl")
)

xgb_model = joblib.load(
    os.path.join(BASE_DIR, "../Artifacts/models/xgboost.pkl")
)

# Utility Functions
def convert_age_to_group(age):
    age = int(age)
    if 18 <= age <= 25:
        return "18-25"
    elif 26 <= age <= 35:
        return "26-35"
    elif 36 <= age <= 45:
        return "36-45"
    elif 46 <= age <= 55:
        return "46-55"
    elif 56 <= age <= 70:
        return "56-70"
    else:
        return "70+"


def compute_cf_ab_score(frequency, awareness):
    freq_map = {"0-2 times": 1, "3-4 times": 2, "5-7 times": 3}
    aware_map = {"0 to 1": 1, "2 to 4": 2, "above 4": 3}

    f = freq_map[frequency]
    a = aware_map[awareness]

    score = f / (f + a)
    return round(score, 2)


def compute_zas_score(zone, income):
    zone = zone.lower().strip()
    income = income.lower().strip()

    zone_map = {
        "rural": 1,
        "semi-urban": 2,
        "urban": 3,
        "metro": 4,
    }

    income_map = {
        "not reported": 0,
        "<10l": 1,
        "10l - 15l": 2,
        "16l - 25l": 3,
        "26l - 35l": 4,
        "> 35l": 5
    }

    return zone_map[zone] * income_map[income]


def compute_bsi(current_brand, reason):
    reason = reason.lower().strip()
    current_brand = current_brand.lower().strip()

    return int(
        (current_brand != "established")
        and (reason in ["price", "quality"])
    )



# FINAL COLUMN ORDER
final_columns = [
    'income_levels', 'consume_frequency(weekly)', 'preferable_consumption_size',
    'health_concerns', 'age_group', 'cf_ab_score', 'zas_score', 'bsi',
    'gender_M', 'zone_rural', 'zone_semi-urban', 'zone_urban',
    'occupation_retired', 'occupation_student', 'occupation_working professional',
    'current_brand_newcomer', 'awareness_of_other_brands_2 to 4',
    'awareness_of_other_brands_above 4',
    'reasons_for_choosing_brands_brand reputation',
    'reasons_for_choosing_brands_price', 'reasons_for_choosing_brands_quality',
    'flavor_preference_Traditional', 'purchase_channel_Retail Store',
    'packaging_preference_Premium', 'packaging_preference_Simple',
    'typical_consumption_situations_Casual (eg. At home)',
    'typical_consumption_situations_Social (eg. Parties)'
]

# Preprocessing Pipeline
def preprocess_input(user_input):

    df = pd.DataFrame([user_input])
    df["income_levels"] = (df["income_levels"].str.lower().str.replace("l", "l").str.strip())
    df["zone"] = df["zone"].str.lower().str.strip()
    df["current_brand"] = df["current_brand"].str.lower().str.strip()
    df["reasons_for_choosing_brands"] = df["reasons_for_choosing_brands"].str.lower().str.strip()
    df["awareness_of_other_brands"] = df["awareness_of_other_brands"].str.strip()

    # Compute Derived Features
    df["cf_ab_score"] = compute_cf_ab_score(
        df["consume_frequency(weekly)"].iloc[0],
        df["awareness_of_other_brands"].iloc[0]
    )

    df["zas_score"] = compute_zas_score(
        df["zone"].iloc[0],
        df["income_levels"].iloc[0]
    )

    df["bsi"] = compute_bsi(
        df["current_brand"].iloc[0],
        df["reasons_for_choosing_brands"].iloc[0]
    )

    # Label Encoding
    for col, encoder in label_encoders.items():
        df[col] = encoder.transform(df[col])

    # OneHot Encoding
    onehot_cols = [
        'gender', 'zone', 'occupation', 'current_brand',
        'awareness_of_other_brands', 'reasons_for_choosing_brands',
        'flavor_preference', 'purchase_channel', 'packaging_preference',
        'typical_consumption_situations'
    ]

    ohe_vals = ohe.transform(df[onehot_cols])
    ohe_features = ohe.get_feature_names_out(onehot_cols)

    df[ohe_features] = ohe_vals
    df.drop(columns=onehot_cols, inplace=True)

    # Add missing columns
    for col in final_columns:
        if col not in df:
            df[col] = 0

    df = df[final_columns]

    # Scaling
    df_scaled = scaler.transform(df)

    return df_scaled


# Prediction
def predict_price_range(user_input):

    processed = preprocess_input(user_input)

    pred = xgb_model.predict(processed)[0]

    return price_le.inverse_transform([pred])[0]

if __name__ == "__main__":
    sample = {
        "age": 30,
        "gender": "M",
        "zone": "Urban",
        "occupation": "working professional",
        "income_levels": "16l - 25l",
        "consume_frequency(weekly)": "3-4 times",
        "preferable_consumption_size": "Medium (500 ml)",
        "health_concerns": "Low (Not very concerned)",
        "current_brand": "newcomer",
        "awareness_of_other_brands": "2 to 4",
        "reasons_for_choosing_brands": "price",
        "flavor_preference": "Traditional",
        "purchase_channel": "Retail Store",
        "packaging_preference": "Simple",
        "typical_consumption_situations": "Casual (eg. At home)"
    }

    print("Prediction:", predict_price_range(sample))
