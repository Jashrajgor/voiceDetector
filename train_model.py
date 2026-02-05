import librosa
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    features = [
        np.mean(librosa.feature.spectral_flatness(y=y)),
        np.mean(librosa.feature.zero_crossing_rate(y)),
        np.mean(librosa.feature.mfcc(y=y, sr=sr))
    ]
    return np.array(features)

# Example dummy training data
X = []
y = []

# Human voices
for _ in range(20):
    X.append(np.random.rand(3))
    y.append(0)

# AI voices
for _ in range(20):
    X.append(np.random.rand(3) + 0.3)
    y.append(1)

model = RandomForestClassifier()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))
print("Model trained & saved")
