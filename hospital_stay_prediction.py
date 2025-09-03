import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score,
                             roc_curve, precision_recall_curve)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

# Устанавливаем стиль для графиков
plt.style.use('default')
sns.set_palette("husl")


class DiabetesPredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.optimal_threshold = 0.5
        self.models_results = {}

    def load_data(self, file_path):
        """Загрузка и предварительный просмотр данных из CSV"""
        print("Загрузка данных из CSV...")
        try:
            self.df = pd.read_csv(file_path)
            self.df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = self.df[
                ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
            self.df = self.df.fillna(self.df.median())

            print(f"✅ Данные успешно загружены. Размер датасета: {self.df.shape}")
            print("\n📊 Первые 5 строк:")
            print(self.df.head())
            print("\n📈 Статистика:")
            print(self.df.describe())
            print(f"\n🎯 Распределение целевой переменной:")
            print(self.df['Outcome'].value_counts())
            print(f"Соотношение: {self.df['Outcome'].value_counts(normalize=True)}")
            return self.df
        except Exception as e:
            print(f"❌ Ошибка при загрузке данных: {e}")
            return None

    def explore_data(self):
        """Визуальный анализ данных"""
        print("Визуальный анализ данных...")
        try:
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 2, 1)
            sns.histplot(data=self.df, x='Age', hue='Outcome', multiple='stack', bins=30)
            plt.title('Распределение возраста по исходу')
            plt.xlabel('Возраст')
            plt.ylabel('Количество')

            plt.subplot(1, 2, 2)
            sns.boxplot(x='Outcome', y='Glucose', data=self.df)
            plt.title('Уровень глюкозы в зависимости от исхода')
            plt.xlabel('Исход (0: нет, 1: есть диабет)')
            plt.ylabel('Уровень глюкозы')

            plt.tight_layout()
            plt.savefig('diabetes_data_analysis.png')
            plt.show()
            print("✅ Визуальный анализ завершен. График сохранен как diabetes_data_analysis.png")
        except Exception as e:
            print(f"❌ Ошибка при визуализации данных: {e}")

    def prepare_data(self):
        """Подготовка данных для моделирования"""
        print("Подготовка данных...")
        try:
            target_col = 'Outcome'
            y = self.df[target_col]
            X = self.df.drop(columns=[target_col])
            self.feature_names = X.columns.tolist()
            numerical_features = X.columns.tolist()

            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_features)
                ])

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)

            print(f"✅ Данные подготовлены. Размер обучающей выборки: {X_train.shape}")
            return X_train, X_test, y_train, y_test, self.feature_names
        except Exception as e:
            print(f"❌ Ошибка при подготовке данных: {e}")
            return None, None, None, None, None

    def train_models(self, X_train, X_test, y_train, y_test):
        """Обучение и оценка нескольких моделей"""
        print("Обучение базовых моделей...")

        models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')  # <-- ИЗМЕНЕНИЕ ЗДЕСЬ
        }

        for name, model in models.items():
            print(f"\n▶️ Обучение и оценка {name}...")
            pipeline = Pipeline(steps=[('preprocessor', self.preprocessor),
                                       ('classifier', model)])
            pipeline.fit(X_train, y_train)
            self.evaluate_model(pipeline, X_test, y_test, model_name=name)
            self.models_results[name] = pipeline

        return self.models_results

    def tune_best_model(self, X_train, y_train):
        """Настройка гиперпараметров лучшей модели (XGBoost)"""
        print("\nНастройка лучшей модели (XGBoost)...")

        pipeline = ImbPipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))  # <-- ИЗМЕНЕНИЕ ЗДЕСЬ
        ])

        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.1, 0.2]
        }

        search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
        search.fit(X_train, y_train)

        self.model = search.best_estimator_
        self.optimal_threshold = self.find_optimal_threshold(self.model, X_train, y_train)

        print("✅ Настройка завершена.")
        print(f"   > Лучшие параметры: {search.best_params_}")
        print(f"   > Лучший F1-Score: {search.best_score_:.4f}")
        print(f"   > Оптимальный порог: {self.optimal_threshold:.4f}")

    def train_stacking_ensemble(self, X_train, y_train):
        """Создание и обучение ансамбля стекинга"""
        print("\n▶️ Создание и обучение ансамбля стекинга...")

        # Базовые классификаторы
        estimators = [
            ('lr', LogisticRegression(random_state=42)),
            ('rf', RandomForestClassifier(random_state=42)),
            ('xgb', XGBClassifier(random_state=42, eval_metric='logloss'))  # <-- ИЗМЕНЕНИЕ ЗДЕСЬ
        ]

        # Мета-классификатор (финальная модель)
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            cv=5,
            n_jobs=-1
        )

        pipeline = ImbPipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', stacking_clf)
        ])

        pipeline.fit(X_train, y_train)
        self.model = pipeline
        self.optimal_threshold = self.find_optimal_threshold(self.model, X_train, y_train)

        print("✅ Ансамбль стекинга обучен.")
        print(f"   > Оптимальный порог для ансамбля: {self.optimal_threshold:.4f}")

    def find_optimal_threshold(self, model, X_train, y_train):
        """Находит оптимальный порог для классификации"""
        y_scores = model.predict_proba(X_train)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        f1_scores = np.nan_to_num(f1_scores)
        optimal_threshold = thresholds[np.argmax(f1_scores)]
        return optimal_threshold

    def evaluate_model(self, model, X_test, y_test, model_name="Модель"):
        """Комплексная оценка модели"""
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        optimal_threshold = self.find_optimal_threshold(model, X_test, y_test)
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)

        print(f"\n▶️ Отчет по классификации для {model_name}:")
        print(classification_report(y_test, y_pred))

        print(f"▶️ Матрица ошибок для {model_name}:")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Матрица ошибок для {model_name}')
        plt.xlabel('Предсказано')
        plt.ylabel('Истина')
        plt.savefig(f'diabetes_{model_name.replace(" ", "_")}_confusion_matrix.png')
        plt.show()

        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"   > Точность: {accuracy_score(y_test, y_pred):.4f}")
        print(f"   > F1-Score: {f1_score(y_test, y_pred):.4f}")
        print(f"   > ROC-AUC: {roc_auc:.4f}")
        print(f"   > Оптимальный порог: {optimal_threshold:.4f}")

        return {'roc_auc': roc_auc, 'model': model}

    def final_evaluation(self, X_test, y_test):
        """Оценка всех моделей на одном графике"""
        print("\n▶️ Построение ROC-кривой для всех моделей...")
        plt.figure(figsize=(8, 6))

        for name, model_pipeline in self.models_results.items():
            y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

        # Добавляем финальный ансамбль
        final_ensemble_proba = self.model.predict_proba(X_test)[:, 1]
        final_fpr, final_tpr, _ = roc_curve(y_test, final_ensemble_proba)
        final_auc = roc_auc_score(y_test, final_ensemble_proba)
        plt.plot(final_fpr, final_tpr, 'k--', label=f'Ансамбль (AUC = {final_auc:.2f})', linewidth=2.5)

        plt.plot([0, 1], [0, 1], 'r--', label='Случайная модель')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-кривая для всех моделей')
        plt.legend()
        plt.savefig('diabetes_all_models_roc_curve.png')
        plt.show()
        print("✅ Оценка всех моделей завершена. График сохранен как diabetes_all_models_roc_curve.png")

    def feature_importance(self, feature_names):
        """Визуализация важности признаков"""
        print("\nВизуализация важности признаков...")
        if self.model and hasattr(self.model.named_steps['classifier'], 'final_estimator_'):
            # Извлекаем важность признаков из финального оценщика стекинга
            final_estimator = self.model.named_steps['classifier'].final_estimator_
            if hasattr(final_estimator, 'coef_'):
                importances = final_estimator.coef_[0]
            elif hasattr(final_estimator, 'feature_importances_'):
                importances = final_estimator.feature_importances_
            else:
                print("❌ Важность признаков не может быть определена для финального оценщика.")
                return

            feature_importance_df = pd.DataFrame({
                'Feature': ['LR', 'RF', 'XGB'],
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)

            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
            plt.title('Важность признаков для ансамбля')
            plt.xlabel('Важность')
            plt.ylabel('Признак')
            plt.tight_layout()
            plt.savefig('diabetes_ensemble_feature_importance.png')
            plt.show()
            print("✅ Важность признаков ансамбля сохранена как diabetes_ensemble_feature_importance.png")
        else:
            print("❌ Важность признаков не может быть определена для данной модели.")

    def save_model(self):
        """Сохранение обученной модели"""
        print("\nСохранение модели...")
        joblib.dump(self.model, 'diabetes_stacking_ensemble.pkl')
        print("✅ Модель сохранена как diabetes_stacking_ensemble.pkl")


def main():
    predictor = DiabetesPredictor()
    file_path = 'diabetes.csv'

    if os.path.exists(file_path):
        print("ℹ️ Используем CSV файл")
        df = predictor.load_data(file_path)
    else:
        print(
            "❌ Файл diabetes.csv не найден. Пожалуйста, убедитесь, что он находится в той же директории, что и скрипт.")
        return

    if df is None or df.empty:
        print("❌ Не удалось загрузить данные")
        return

    predictor.explore_data()
    X_train, X_test, y_train, y_test, feature_names = predictor.prepare_data()

    if X_train is not None:
        # Обучение и оценка базовых моделей
        results = predictor.train_models(X_train, X_test, y_train, y_test)

        # Настройка лучшей модели (XGBoost) - опционально, но полезно
        predictor.tune_best_model(X_train, y_train)

        # Обучение ансамбля
        predictor.train_stacking_ensemble(X_train, y_train)

        # Финальная оценка ансамбля
        print("\n" + "=" * 50)
        print("ФИНАЛЬНАЯ ОЦЕНКА АНСАМБЛЯ СТЕКИНГА")
        print("=" * 50)
        predictor.evaluate_model(predictor.model, X_test, y_test, model_name="Ансамбль")
        predictor.final_evaluation(X_test, y_test)
        predictor.save_model()

    print("\n" + "=" * 50)
    print("✅ ПРОЦЕСС УСПЕШНО ЗАВЕРШЕН")
    print("=" * 50)
    print("📁 Созданы файлы:")
    print("- diabetes_data_analysis.png")
    print("- diabetes_Logistic_Regression_confusion_matrix.png")
    print("- diabetes_Random_Forest_confusion_matrix.png")
    print("- diabetes_XGBoost_confusion_matrix.png")
    print("- diabetes_Ансамбль_confusion_matrix.png")
    print("- diabetes_all_models_roc_curve.png")
    print("- diabetes_stacking_ensemble.pkl")


if __name__ == "__main__":
    main()