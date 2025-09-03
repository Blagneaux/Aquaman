from sklearn.inspection import permutation_importance
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import shap

# Load your labeled vortex features
df = pd.read_csv("vortex_features.csv")

# Filter only match and semi match labels
df_filtered = df[df["label"].isin(["match", "semi match"])].copy()
df_filtered["fish"] = df_filtered["fish"].astype(int)

# Select features and target
X = df_filtered.drop(columns=["haato", "haachama", "sensor", "fish", "vortex_id", "label"])
X = (X-X.min())/(X.max()-X.min())
y = df_filtered["fish"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# (Optional) Standardization — often unnecessary for tree models, but you can try it
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# # Define the large grid
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'max_depth': [3, 5, 7],
#     'min_samples_split': [2, 5],
#     'subsample': [1.0, 0.8]
# }

# # Initialize model
# gb = GradientBoostingClassifier(random_state=42)

# # Grid search with 5-fold cross-validation
# grid_search_gradBoost = GridSearchCV(
#     estimator=gb,
#     param_grid=param_grid,
#     cv=5,
#     scoring='accuracy',
#     n_jobs=-1,
#     verbose=2  # Increase verbosity to monitor progress
# )

# # Fit the model
# grid_search_gradBoost.fit(X, y)

# # Define hyperparameter grid
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [None, 5, 10],
#     'min_samples_split': [2, 5],
#     'max_features': ['sqrt', 'log2']
# }

# # Run grid search with 5-fold CV
# grid_search_randomForest = GridSearchCV(
#     estimator=RandomForestClassifier(random_state=42),
#     param_grid=param_grid,
#     cv=5,
#     scoring='accuracy',
#     n_jobs=-1,
#     verbose=2
# )

# grid_search_randomForest.fit(X_train, y_train)

# # # Show the best results
# # print("✅ Best Parameters Found:")
# # print(grid_search_gradBoost.best_params_)
# # print(f"✅ Best Cross-Validation Accuracy: {grid_search_gradBoost.best_score_:.4f}")

# print("✅ Best Parameters:", grid_search_randomForest.best_params_)
# print("✅ Best Cross-Validation Accuracy:", grid_search_randomForest.best_score_)

# Train model
model = RandomForestClassifier(
    n_estimators=200, max_depth=None, max_features='sqrt', min_samples_split=2, random_state=42
)


model.fit(X_train, y_train)

from sklearn.metrics import classification_report
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)

print(report)


# explainer = shap.Explainer(model, X)
# shap_values = explainer(X)

# shap.plots.beeswarm(shap_values[:, :, 1])  # Class index 1


# # Classifier for match and semi match vs no match
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report
# from imblearn.over_sampling import SMOTE

# # Load data
# df = pd.read_csv("vortex_features.csv")

# # Create binary label: match or semi match = 1, no match = 0
# df["label_binary"] = df["label"].map(lambda x: 1 if x in ["match", "semi match"] else 0)

# # Define features and target
# X = df.drop(columns=["haato", "haachama", "sensor", "fish", "vortex_id", "label", "label_binary"])
# y = df["label_binary"]

# # Apply SMOTE
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X, y)

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
# )

# # Train Random Forest
# rf = RandomForestClassifier(random_state=42)
# rf.fit(X_train, y_train)

# # Evaluate
# y_pred = rf.predict(X_test)
# print("Classification Report:\n")
# print(classification_report(y_test, y_pred))

# explainer = shap.Explainer(rf, X)
# shap_values = explainer(X)

# shap.plots.beeswarm(shap_values[:, :, 1])  # Class index 1
