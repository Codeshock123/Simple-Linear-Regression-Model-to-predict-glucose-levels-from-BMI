# Simple-Linear-Regression-Model-to-predict-glucose-levels-from-BMI
🧩 Problem Statement: Can BMI predict blood glucose levels using linear regression? This mini-project explores whether BMI alone is enough to estimate a person's glucose levels using simple linear regression.

---

📂 Dataset
The dataset used is the PIMA Indians Diabetes Dataset (publicly available on Kaggle).[https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database]
  Total rows: 768
  Columns used: BMI, Glucose
  Zero values in BMI and Glucose were removed for accuracy.

---

  🧠 Code Explanation
  
  1. Importing libraries:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
```

2. Loading the dataset
```
df = pd.read_csv('/content/sample_data/diabetes.csv')
df.head()
```

3. Summary statistics
```
df.describe()
```

4. Checking for missing values
```
df.isnull().sum()
```

5. Removing invalid zero entries
```
df = df[(df["BMI"] > 0) & (df["Glucose"] > 0)]
```

6. Checking zero-value counts
```
(df == 0).sum()
```

7. Updated statistics
```
df.describe()
```

8. Scatter plot (BMI vs Glucose)
```
plt.scatter(df['BMI', df['Glucose'])
plt.xlabel('BMI')
plt.ylabel('Glucose')
plt.title('BMI vs Glucose')
plt.show()

9. Correlation
```
df[['BMI', 'Glucose']].corr()
```

10. Feature selection
```
X= df[['BMI']]
Y= df[['Glucose']]
```

11. Train-test split
```
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size= 0.2, random_state= 42)
```

12. Training the Linear Regression Model
```
model= LinearRegression()
model.fit(X_train, y_train)
```

13. Making predictions
```
y_pred= mode.predict(X_test)
```

14. Model Evaluation
```
print("R2 score", r2_score(y_test, y_pred))
print("MSE", mean_squared_error(y_test, y_pred))
```

15. Plotting regression line
```
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred)
plt.xlabel("BMI")
plt.ylabel("Glucose")
plt.title("Linear Regression Line")
plt.show()
```

16. Coefficient and Intercept
```
model.coef_
model.intercept_
```

---

📊 Metrics

| Metric                     | Value    |
|---------------------------|----------|
| R² (R Square)             | 0.0326   |
| Mean Squared Error (MSE)  | 950.8    |
| Slope (Coefficient)       | 1.03     |
| Intercept                 | 87.67    |

---

📈 Charts
The project includes:
  Scatter plot: BMI vs Glucose
  Regression line plot
These help visualize the very weak relationship between BMI and Glucose.

---

📝 Interpretation

The Linear Regression model performs poorly when using BMI to predict glucose levels. The R² score is extremely low (3.26%), indicating BMI explains almost none of the glucose variation. The high MSE (950.8) shows that predictions are far off from real glucose values.

The regression coefficient (1.03) shows BMI has minimal linear influence. The intercept (87.67) is only a mathematical baseline and does not represent a real glucose value.

---

Conclusion:
BMI and Glucose have a weak relationship, and simple linear regression is not effective for predicting glucose using BMI alone.

---

🚀 Future Work (Short)

Add more features such as Age, Insulin, BloodPressure, etc.
Try multivariate regression models.
Compare performance with Random Forest or XGBoost.

---

