import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score,
                             roc_curve, precision_recall_curve)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline  # Важно!
import joblib
import warnings

warnings.filterwarnings('ignore')

# Устанавливаем стиль для графиков
plt.style.use('default')
sns.set_palette("husl")


class HospitalStayPredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.optimal_threshold = 0.5

    def load_data(self, file_path):
        """Загрузка и предварительный просмотр данных"""
        print("Загрузка данных...")
        self.df = pd.read_csv(file_path)

        # Создаем новые признаки на основе имеющихся
        self.df['bp_heart_ratio'] = self.df['blood_pressure'] / self.df['heart_rate']
        self.df['age_disease_interaction'] = self.df['age'] * self.df['previous_diseases']
        self.df['temp_oxygen_interaction'] = self.df['temperature'] * self.df['oxygen_saturation']
        self.df = self.df.replace([np.inf, -np.inf], np.nan).fillna(0)

        print(f"Размер датасета: {self.df.shape}")
        print("\nПервые 5 строк:")
        print(self.df.head())
        print("\nСтатистика:")
        print(self.df.describe())

        return self.df

    def explore_data(self):
        """Разведочный анализ данных"""
        print("\n=== РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ ===\n")

        fig = plt.figure(figsize=(20, 15))

        # 1. Распределение длительности госпитализации
        ax1 = plt.subplot(2, 3, 1)
        sns.histplot(self.df['length_of_stay'], bins=20, kde=True, ax=ax1)
        ax1.set_title('Распределение длительности госпитализации', fontsize=14)
        ax1.set_xlabel('Дни')
        ax1.set_ylabel('Количество пациентов')

        # 2. Соотношение коротких/длительных госпитализаций
        ax2 = plt.subplot(2, 3, 2)
        stay_counts = self.df['long_stay'].value_counts()
        colors = ['#66b3ff', '#ff6666']
        wedges, texts, autotexts = ax2.pie(stay_counts.values,
                                           labels=['Короткий (<5 дн)', 'Длительный (5+ дн)'],
                                           autopct='%1.1f%%', colors=colors,
                                           startangle=90)
        ax2.set_title('Соотношение коротких/длительных госпитализаций', fontsize=14)

        # 3. Корреляционная матрица
        ax3 = plt.subplot(2, 3, 3)
        numeric_df = self.df.select_dtypes(include=[np.number])
        numeric_features = numeric_df.drop(['length_of_stay', 'long_stay'], axis=1, errors='ignore')
        correlation_matrix = numeric_features.corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                    square=True, ax=ax3, mask=mask, fmt='.2f')
        ax3.set_title('Матрица корреляций числовых признаков', fontsize=14)

        # 4. Возраст vs Длительность пребывания
        ax4 = plt.subplot(2, 3, 4)
        scatter = ax4.scatter(self.df['age'], self.df['length_of_stay'],
                              c=self.df['long_stay'], cmap='viridis', alpha=0.6)
        ax4.set_xlabel('Возраст')
        ax4.set_ylabel('Длительность пребывания (дни)')
        ax4.set_title('Возраст vs Длительность пребывания', fontsize=14)
        legend = ax4.legend(*scatter.legend_elements(), title="Тип пребывания")
        ax4.add_artist(legend)

        # 5. Сатурация кислорода по группам
        ax5 = plt.subplot(2, 3, 5)
        boxplot = sns.boxplot(x='long_stay', y='oxygen_saturation', data=self.df, ax=ax5)
        ax5.set_xlabel('Тип госпитализации')
        ax5.set_ylabel('Сатурация кислорода (%)')
        ax5.set_title('Сатурация кислорода по группам', fontsize=14)
        ax5.set_xticklabels(['Короткий', 'Длительный'])

        # 6. Тип госпитализации
        ax6 = plt.subplot(2, 3, 6)
        emergency_cross = pd.crosstab(self.df['emergency_admission'], self.df['long_stay'])
        emergency_cross.columns = ['Короткий', 'Длительный']
        emergency_cross.index = ['Плановая', 'Экстренная']
        emergency_cross.plot(kind='bar', ax=ax6, color=['#66b3ff', '#ff6666'])
        ax6.set_title('Тип госпитализации vs Длительность', fontsize=14)
        ax6.set_xlabel('Тип госпитализации')
        ax6.set_ylabel('Количество пациентов')
        ax6.legend(title='Длительность')

        plt.tight_layout()
        plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("График анализа данных сохранен в 'data_analysis.png'")

        print("\nСтатистика по группам:")
        short_stay = self.df[self.df['long_stay'] == 0]
        long_stay = self.df[self.df['long_stay'] == 1]
        print(f"Короткая госпитализация (<5 дней): {len(short_stay)} пациентов")
        print(f"Длительная госпитализация (5+ дней): {len(long_stay)} пациентов")
        print(f"Средняя длительность: {self.df['length_of_stay'].mean():.2f} дней")
        print("\nПропущенные значения:")
        print(self.df.isnull().sum())

    def prepare_data(self):
        """Подготовка данных для обучения"""
        print("\n=== ПОДГОТОВКА ДАННЫХ ===\n")

        X = self.df.drop(['length_of_stay', 'long_stay'], axis=1)
        y = self.df['long_stay']

        numeric_features = ['age', 'blood_pressure', 'heart_rate',
                            'temperature', 'oxygen_saturation',
                            'previous_diseases', 'treatment_intensity',
                            'bp_heart_ratio', 'age_disease_interaction', 'temp_oxygen_interaction']
        categorical_features = ['gender']

        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Обучающая выборка: {X_train.shape}")
        print(f"Тестовая выборка: {X_test.shape}")
        print(f"Соотношение классов в обучающей: {y_train.value_counts(normalize=True).to_dict()}")
        print(f"Соотношение классов в тестовой: {y_test.value_counts(normalize=True).to_dict()}")

        return X_train, X_test, y_train, y_test, X.columns

    def train_models(self, X_train, X_test, y_train, y_test):
        """Обучение и сравнение нескольких моделей"""
        print("\n=== ОБУЧЕНИЕ МОДЕЛЕЙ ===\n")

        scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

        # Модели с пайплайнами, включающими SMOTE
        models = {
            'Logistic Regression': ImbPipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('smote', SMOTE(random_state=42)),
                ('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
            ]),
            'Random Forest': ImbPipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('smote', SMOTE(random_state=42)),
                ('classifier', RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'))
            ]),
            'XGBoost': ImbPipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('smote', SMOTE(random_state=42)),
                ('classifier', XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=scale_pos_weight))
            ])
        }

        results = {}

        for name, pipeline in models.items():
            print(f"Обучение {name}...")
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)

            results[name] = {
                'model': pipeline,
                'accuracy': accuracy,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }

            print(f"{name}: Accuracy = {accuracy:.3f}, F1 = {f1:.3f}, ROC-AUC = {roc_auc:.3f}")

        # Стекинг с SMOTE
        print("Обучение Stacking Classifier...")

        base_models = [
            ('lr', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')),
            ('rf', RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')),
            ('xgb', XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=scale_pos_weight))
        ]

        stacking_pipeline = ImbPipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', StackingClassifier(
                estimators=base_models,
                final_estimator=LogisticRegression(class_weight='balanced'),
                cv=5
            ))
        ])

        stacking_pipeline.fit(X_train, y_train)
        y_pred_stack = stacking_pipeline.predict(X_test)
        y_pred_proba_stack = stacking_pipeline.predict_proba(X_test)[:, 1]

        accuracy_stack = accuracy_score(y_test, y_pred_stack)
        f1_stack = f1_score(y_test, y_pred_stack)
        roc_auc_stack = roc_auc_score(y_test, y_pred_proba_stack)

        results['Stacking'] = {
            'model': stacking_pipeline,
            'accuracy': accuracy_stack,
            'f1_score': f1_stack,
            'roc_auc': roc_auc_stack,
            'y_pred': y_pred_stack,
            'y_pred_proba': y_pred_proba_stack
        }

        print(f"Stacking: Accuracy = {accuracy_stack:.3f}, F1 = {f1_stack:.3f}, ROC-AUC = {roc_auc_stack:.3f}")

        best_model_name = max(results.items(), key=lambda x: x[1]['f1_score'])[0]
        self.model = results[best_model_name]['model']

        print(f"\nЛучшая модель: {best_model_name}")
        print(f"F1-score: {results[best_model_name]['f1_score']:.3f}")

        return results

    def tune_best_model(self, X_train, y_train):
        """Настройка гиперпараметров лучшей модели"""
        print("\n=== НАСТРОЙКА ГИПЕРПАРАМЕТРОВ ЛУЧШЕЙ МОДЕЛИ ===\n")

        # Используем пайплайн с GridSearch
        pipeline = ImbPipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
        ])

        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2]
        }

        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=3,  # Уменьшил для скорости
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        print(f"Лучшие параметры: {grid_search.best_params_}")
        print(f"Лучший F1-score на кросс-валидации: {grid_search.best_score_:.3f}")

        self.model = grid_search.best_estimator_

        return grid_search

    def evaluate_model(self, results, X_test, y_test):
        """Детальная оценка лучшей модели"""
        print("\n=== ОЦЕНКА МОДЕЛИ ===\n")

        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        print("Classification Report (Порог 0.5):")
        print(classification_report(y_test, y_pred, target_names=['Короткий', 'Длительный']))

        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        f1_scores = (2 * precision * recall) / (precision + recall + 1e-7)
        optimal_threshold_idx = np.argmax(f1_scores)
        self.optimal_threshold = thresholds[optimal_threshold_idx]

        print(f"\nОптимальный порог классификации: {self.optimal_threshold:.3f}")
        y_pred_optimal = (y_pred_proba >= self.optimal_threshold).astype(int)

        print("\nClassification Report (Оптимальный порог):")
        print(classification_report(y_test, y_pred_optimal, target_names=['Короткий', 'Длительный']))

        # Визуализация
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Матрица ошибок (порог 0.5)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                    xticklabels=['Короткий', 'Длительный'],
                    yticklabels=['Короткий', 'Длительный'])
        ax1.set_title('Матрица ошибок (Порог 0.5)', fontsize=14)

        # Матрица ошибок (оптимальный порог)
        cm_optimal = confusion_matrix(y_test, y_pred_optimal)
        sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Blues', ax=ax2,
                    xticklabels=['Короткий', 'Длительный'],
                    yticklabels=['Короткий', 'Длительный'])
        ax2.set_title(f'Матрица ошибок (Порог {self.optimal_threshold:.2f})', fontsize=14)

        # ROC-кривая
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('ROC-кривая', fontsize=14)
        ax3.legend(loc='lower right')

        # Precision-Recall кривая
        ax4.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
        ax4.set_xlabel('Recall')
        ax4.set_ylabel('Precision')
        ax4.set_title('Precision-Recall кривая', fontsize=14)
        ax4.legend(loc='lower left')

        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("График оценки модели сохранен в 'model_evaluation.png'")

    def feature_importance(self, feature_names):
        """Анализ важности признаков"""
        print("\n=== ВАЖНОСТЬ ПРИЗНАКОВ ===\n")

        try:
            classifier = self.model.named_steps['classifier']
            preprocessor = self.model.named_steps['preprocessor']

            numeric_features = preprocessor.named_transformers_['num'].get_feature_names_out()
            categorical_features = preprocessor.named_transformers_['cat'].get_feature_names_out(['gender'])
            all_features = list(numeric_features) + list(categorical_features)

            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
                feature_imp = pd.DataFrame({'feature': all_features, 'importance': importances})
                feature_imp = feature_imp.sort_values('importance', ascending=True)

                plt.figure(figsize=(12, 8))
                bars = plt.barh(feature_imp['feature'], feature_imp['importance'])
                plt.xlabel('Важность признака')
                plt.title('Важность признаков в модели', fontsize=16)
                plt.grid(axis='x', alpha=0.3)

                for bar in bars:
                    width = bar.get_width()
                    plt.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                             f'{width:.4f}', ha='left', va='center')

                plt.tight_layout()
                plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
                plt.close()

                print("График важности признаков сохранен в 'feature_importance.png'")
                print("\nТоп-10 самых важных признаков:")
                print(feature_imp.nlargest(10, 'importance')[['feature', 'importance']].to_string(index=False))

            elif hasattr(classifier, 'coef_'):
                coef = classifier.coef_[0]
                feature_imp = pd.DataFrame({'feature': all_features, 'coefficient': coef})
                feature_imp = feature_imp.sort_values('coefficient', key=abs, ascending=False)
                print("\nКоэффициенты логистической регрессии (топ-10):")
                print(feature_imp.head(10)[['feature', 'coefficient']].to_string(index=False))
            else:
                print("Данная модель не поддерживает анализ важности признаков")

        except Exception as e:
            print(f"Ошибка при анализе важности признаков: {e}")

    def predict_with_threshold(self, X, threshold=None):
        """Предсказание с использованием кастомного порога"""
        if threshold is None:
            threshold = self.optimal_threshold

        y_pred_proba = self.model.predict_proba(X)[:, 1]
        return (y_pred_proba >= threshold).astype(int)

    def save_model(self, file_path='hospital_stay_model.pkl'):
        """Сохранение модели"""
        model_data = {
            'model': self.model,
            'optimal_threshold': self.optimal_threshold,
            'preprocessor': self.preprocessor
        }
        joblib.dump(model_data, file_path)
        print(f"Модель сохранена в {file_path}")

    def load_model(self, file_path='hospital_stay_model.pkl'):
        """Загрузка модели"""
        model_data = joblib.load(file_path)
        self.model = model_data['model']
        self.optimal_threshold = model_data['optimal_threshold']
        self.preprocessor = model_data['preprocessor']
        print(f"Модель загружена из {file_path}")


def main():
    """Основная функция"""
    try:
        predictor = HospitalStayPredictor()
        df = predictor.load_data('synthetic_hospital_data.csv')
        predictor.explore_data()
        X_train, X_test, y_train, y_test, feature_names = predictor.prepare_data()
        results = predictor.train_models(X_train, X_test, y_train, y_test)
        predictor.tune_best_model(X_train, y_train)
        predictor.evaluate_model(results, X_test, y_test)
        predictor.feature_importance(feature_names)
        predictor.save_model()

        print("\n✅ ПРОЦЕСС УСПЕШНО ЗАВЕРШЕН")
        print("Созданы файлы:")
        print("- data_analysis.png - Анализ данных")
        print("- model_evaluation.png - Оценка модели")
        print("- feature_importance.png - Важность признаков")
        print("- hospital_stay_model.pkl - Сохраненная модель")
        print(f"\nОптимальный порог для предсказаний: {predictor.optimal_threshold:.3f}")

    except Exception as e:
        print(f"❌ Произошла ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()