import pandas as pd
import numpy as np

class ModelSelector:
    def __init__(self, df: pd.DataFrame, target_column: str):
        self.df = df.copy()
        self.target_column = target_column
        self.target = df[target_column]
        self.features = df.drop(columns=[target_column])
        self.categorical_cols = self.features.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_cols = self.features.select_dtypes(include=[np.number]).columns.tolist()
        self.model_suggestions = []

    def check_missing_values(self):
        if self.df.isnull().sum().sum() > 0:
            print("🔹 Missing values detected.")
            self.model_suggestions += ["XGBoost", "LightGBM", "CatBoost"]
        else:
            print("✅ No missing values found.")
        return self.model_suggestions

    def check_high_cardinality_categoricals(self, threshold=50):
        high_card_cols = [col for col in self.categorical_cols if self.df[col].nunique() > threshold]
        if high_card_cols:
            print(f"🔹 High-cardinality categorical columns: {high_card_cols}")
            self.model_suggestions += ["CatBoost", "LightGBM"]
        else:
            print("✅ No high-cardinality categorical features.")
        return self.model_suggestions

    def check_skewed_target(self):
        skewness = self.target.skew()
        print(f"📊 Target skewness: {skewness:.2f}")
        if abs(skewness) > 1:
            self.model_suggestions += ["Random Forest", "XGBoost", "CatBoost"]
        return self.model_suggestions

    def check_multicollinearity(self, threshold=0.9):
        if len(self.numerical_cols) < 2:
            return self.model_suggestions
        corr_matrix = self.df[self.numerical_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        if any(upper.max() > threshold):
            print("🔹 High multicollinearity detected.")
            self.model_suggestions += ["Ridge Regression", "Random Forest", "XGBoost"]
        else:
            print("✅ No severe multicollinearity.")
        return self.model_suggestions

    def check_outliers(self):
        if len(self.numerical_cols) == 0:
            return self.model_suggestions
        z_scores = ((self.df[self.numerical_cols] - self.df[self.numerical_cols].mean()) /
                    self.df[self.numerical_cols].std())
        outlier_count = (np.abs(z_scores) > 3).sum().sum()
        print(f"📈 Outlier count: {outlier_count}")
        if outlier_count > 0.05 * self.df.shape[0] * len(self.numerical_cols):
            self.model_suggestions += ["Random Forest", "XGBoost", "Support Vector Regression"]
        return self.model_suggestions

    def check_data_size(self):
        n_rows = self.df.shape[0]
        print(f"📦 Dataset rows: {n_rows}")
        if n_rows > 100000:
            self.model_suggestions += ["LightGBM"]
        elif n_rows < 5000:
            self.model_suggestions += ["Decision Tree", "Ridge Regression", "Support Vector Regression"]
        return self.model_suggestions

    def check_linear_relationship_heuristic(self):
        numeric_features = self.features.select_dtypes(include=[np.number])
        if numeric_features.shape[1] == 0:
            return self.model_suggestions
        correlations = numeric_features.corrwith(self.target)
        avg_corr = correlations.abs().mean()
        print(f"🔍 Avg feature-target correlation: {avg_corr:.2f}")
        if avg_corr > 0.5:
            self.model_suggestions += ["Linear Regression", "Ridge Regression"]
        return self.model_suggestions

    def check_high_dimensionality_vs_samples(self):
        if len(self.features.columns) / self.df.shape[0] > 1:
            print("⚠️ Feature count > Row count: Prefer regularized models or tree ensembles.")
            self.model_suggestions += ["Ridge Regression", "Lasso Regression", "Random Forest", "XGBoost"]
        return self.model_suggestions

    def check_many_categorical_features(self, threshold=0.3):
        ratio = len(self.categorical_cols) / len(self.features.columns)
        if ratio > threshold:
            print(f"🔸 High categorical feature ratio: {ratio:.2f}")
            self.model_suggestions += ["CatBoost"]
        return self.model_suggestions

    def check_target_variance(self):
        var = self.target.var()
        print(f"📉 Target variance: {var:.4f}")
        if var < 0.01:
            self.model_suggestions += ["Tree-based models (low variance targets handled better)"]
        return self.model_suggestions

    def final_model_suggestions(self):
        self.model_suggestions = []  # Reset
        print("\n🔎 Starting model selection analysis...\n")
        self.check_missing_values()
        self.check_high_cardinality_categoricals()
        self.check_skewed_target()
        self.check_multicollinearity()
        self.check_outliers()
        self.check_data_size()
        self.check_linear_relationship_heuristic()
        self.check_high_dimensionality_vs_samples()
        self.check_many_categorical_features()
        self.check_target_variance()

        if not self.model_suggestions:
            print("⚠️ No specific condition triggered. Using baseline suggestions.")
            self.model_suggestions.append("Start with baseline models: Linear Regression, Decision Tree")

        return sorted(set(self.model_suggestions))


# ----------------- Main Test Block -----------------

if __name__ == "__main__":
    try:
        df = pd.read_csv("data.csv")  # Change to your file name
        print("📋 Loaded Columns:", df.columns.tolist())

        selector = ModelSelector(df, target_column='Performance Index')  # Use exact column name
        models = selector.final_model_suggestions()

        print("\n📌 Suggested Regression Models Based on Dataset:")
        for model in models:
            print("-", model)

    except Exception as e:
        print("❌ Error:", str(e))
