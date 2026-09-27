import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_FILE = Path("iris.csv")
df = pd.read_csv(DATA_FILE)

print("Shape:", df.shape)
print(df.head())
print("\
Class distribution:\
", df["species"].value_counts())

features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X = df[features]
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("\
Accuracy:", round(accuracy_score(y_test, pred), 4))
print("\
Classification Report:\
", classification_report(y_test, pred))
print("Confusion Matrix:\
", confusion_matrix(y_test, pred))

sns.pairplot(df, hue="species")
plt.savefig("iris_pairplot.png", dpi=150)
plt.show()
