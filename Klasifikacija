import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

print("Loading CSV...")

df = pd.read_csv("personality_dataset.csv")
df.head()

df = df.replace(r'^\s*$', np.nan, regex=True)
df = df.replace({"Yes": 1, "No": 0})

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="ignore")


for col in df.columns:
    if df[col].dtype == "object" and col != "Personality":
        print(f"WARNING: Dropping non-numeric column: {col}")
        df = df.drop(columns=[col])


label_encoder = LabelEncoder()
df["Personality"] = label_encoder.fit_transform(df["Personality"])

df = df.fillna(df.mean(numeric_only=True))

print("\nData types after cleaning:")
print(df.dtypes)

print("\nChecking for NaNs...")
print(df.isna().sum())

X = df.drop("Personality", axis=1)
y = df["Personality"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled[:5])


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


model = Sequential([
    Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(16, activation="relu"),
    Dense(2, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

print("\nStarting training...")

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)


def plot_cm_with_accuracy(y_true, y_pred, title_prefix):
    cm = confusion_matrix(y_true, y_pred)
    acc = (y_true == y_pred).mean() * 100
    
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_encoder.classes_
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"{title_prefix} – Accuracy: {acc:.2f}%")
    plt.show()


y_train_pred = np.argmax(model.predict(X_train), axis=1)
plot_cm_with_accuracy(y_train, y_train_pred, "Training Confusion Matrix")


y_test_pred = np.argmax(model.predict(X_test), axis=1)
plot_cm_with_accuracy(y_test, y_test_pred, "Test Confusion Matrix")


plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid()
plt.show()

