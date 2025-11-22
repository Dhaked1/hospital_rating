import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load the dataset
data = pd.read_csv('recommend.csv')

# Replace 'Not Available' with NaN
data.replace('Not Available', np.nan, inplace=True)

# Drop rows where any of the required columns are NaN
data = data.dropna(subset=['H_RECMND_DN', 'H_RECMND_DY', 'H_RECMND_DY', 'H_COMP_1_STAR_RATING'])

# Convert columns to numeric
data['H_RECMND_DN'] = pd.to_numeric(data['H_RECMND_DN'])
data['H_RECMND_DY'] = pd.to_numeric(data['H_RECMND_DY'])
data['H_RECMND_PY'] = pd.to_numeric(data['H_RECMND_PY'])
data['H_COMP_1_STAR_RATING'] = pd.to_numeric(data['H_COMP_1_STAR_RATING'])

# Prepare features (X) and target (y)
X = data[['H_RECMND_DN', 'H_RECMND_DY', 'H_RECMND_PY']].values
y = data['H_RECMND_STAR_RATING'].values

# Adjust labels to be zero-based for classification (1 to 5 -> 0 to 4)
y = y - 1

# Convert target to one-hot encoding for multi-class classification
y = to_categorical(y, num_classes=5)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the deep learning model
model = Sequential([
    Dense(64, activation='relu', input_shape=(3,)),  # Input layer: 3 features
    Dense(32, activation='relu'),                   # Hidden layer
    Dense(16, activation='relu'),                   # Hidden layer
    Dense(5, activation='softmax')                  # Output layer: 5 classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")

# Save the model in Keras format
model.save('hospital_rating_model.keras')

# Optional: Save the scaler for future use (if you need to preprocess new data)
import joblib
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully.")