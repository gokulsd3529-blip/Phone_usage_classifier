import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

# Dataset
data = {
    'screen_time_hours': [1.2, 3.5, 5.0, 0.5, 6.2, 4.5, 2.1, 7.0, 3.8, 1.0],
    'calls_per_day':     [5, 10, 20, 2, 30, 15, 8, 40, 12, 4],
    'messages_per_day':  [20, 50, 120, 10, 200, 90, 40, 300, 80, 15],
    'social_media_hours':[0.5, 2.0, 3.5, 0.2, 4.0, 3.0, 1.0, 5.5, 2.5, 0.7],
    'category': ['Light','Moderate','Heavy','Light','Heavy','Heavy','Moderate','Heavy','Moderate','Light']
}

df = pd.DataFrame(data)

X = df.drop('category', axis=1)
y = df['category']

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500)
model.fit(X_scaled, y)

# Save model & scaler
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(encoder, open("encoder.pkl", "wb"))

print("✅ Model and Scaler saved!")