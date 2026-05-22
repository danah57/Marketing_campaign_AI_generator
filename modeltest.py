from ml.pipeline.inference import InferenceEngine

engine = InferenceEngine()

campaign = {
    "Channel_Used": "Instagram",
    "Campaign_Type": "Search",
    "Audience_age_range": "25-34",
    "Audience_Gender": "Men",
    "Customer_Segment": "Tech Enthusiasts",
    "Location": "New York",
    "Language": "English",
    "Duration": 30,
    "Date": "2024-06-15",
    "Budget": 12000,
}

output = engine.predict_one(
    campaign,
    verbose=False,
    include_shap=True
    
)

print("\n=== SUCCESS ===\n")

print("ROI:")
print(output["stage2_evaluation"]["predicted_roi"])

print("\nSuccess Probability:")
print(output["stage2_evaluation"]["success_probability"])

print("\nVerdict:")
print(output["stage2_evaluation"]["verdict"])

print("\nSHAP:")
print(output["shap_explanation"])
# from pathlib import Path

# path = Path(r"D:\SocialMedia Marketing\AI\campaign_model\ml\artifacts\stage2\stage2_roi_model.pkl")

# print(path.exists())
# print(path.stat().st_size)
# import pickle

# path = r"D:\SocialMedia Marketing\AI\campaign_model\ml\artifacts\stage2\stage2_roi_model.pkl"

# with open(path, "rb") as f:
#     model = pickle.load(f)

# print(type(model))