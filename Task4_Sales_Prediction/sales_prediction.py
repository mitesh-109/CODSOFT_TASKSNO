import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_URL = "https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv"
DATA_FILE = Path("Advertising.csv")

if not DATA_FILE.exists():
    print("Downloading Advertising dataset...")
    df = pd.read_csv(DATA_URL)
    df.to_csv(DATA_FILE, index=False)
else:
    df = pd.read_csv(DATA_FILE)

# Remove an unnamed index column if present.
df = df.loc[:, ~df.columns.str.contains("^Unnamed")].copy()

print("Shape:", df.shape)
print(df.head())
print("\
Summary:\
", df.describe())

features = ["TV", "radio", "newspaper"]
target = "sales"
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred) ** 0.5
r2 = r2_score(y_test, pred)

print("\
MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R2:", round(r2, 4))
print("Coefficients:", dict(zip(features, model.coef_)))
print("Intercept:", model.intercept_)

plt.figure(figsize=(7,5))
sns.scatterplot(x=y_test, y=pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.show()
