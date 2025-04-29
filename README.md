# 🏡 California Housing Price Prediction using Linear Regression

This project demonstrates a simple machine learning pipeline using **Linear Regression** to predict housing prices based on features from the **California Housing dataset**, included in `scikit-learn`.

---

## 📦 Requirements

Make sure the following Python packages are installed:

```bash
pip install pandas scikit-learn
```

---

## 📊 Dataset

We use the [California Housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html), which includes:

- **Features**: Median income, house age, average rooms, etc.
- **Target**: Median house value for California districts (in hundreds of thousands of dollars).

---

## ⚙️ Workflow

1. **Import and Load Data**
   - Load the dataset using `sklearn.datasets.fetch_california_housing`.
   - Convert to a Pandas DataFrame for exploration.

2. **Train-Test Split**
   - Use `train_test_split` to divide data (80% training, 20% testing).

3. **Preprocessing**
   - Apply `StandardScaler` to normalize the features.

4. **Train Model**
   - Fit a `LinearRegression` model on the scaled training data.

5. **Evaluate**
   - Predict housing values using the test set.
   - Use `.score()` to check the model’s performance (R² score).

---

## 📄 Code Highlights

```python
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.DataFrame(housing.target, columns=["MedHouseValue"])

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and predict
model = LinearRegression()
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)

# Evaluate
print("R² score:", model.score(X_test_scaled, y_test))
```

---

## ✅ Output Example

```
R² score: 0.6098
```

This means about 61% of the variance in housing prices can be explained by the model.

---

## 📌 Notes

- You can further improve this model by using more complex algorithms (e.g., Decision Trees, Random Forests).
- Scaling is essential for many ML algorithms, especially when features are in different units.
