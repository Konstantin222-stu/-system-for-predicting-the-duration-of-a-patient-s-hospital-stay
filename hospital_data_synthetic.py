import pandas as pd
import numpy as np



def generate_synthetic_data(n_samples = 1000):
    np.random.seed(42)

    data = {
        'age': np.random.normal(55, 15, n_samples).astype(int),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'blood_pressure': np.random.normal(120, 20, n_samples).astype(int),
        'heart_rate': np.random.normal(75, 12, n_samples).astype(int),
        'temperature': np.random.normal(36.6, 0.8, n_samples),
        'oxygen_saturation': np.random.normal(97, 2, n_samples),
        'emergency_admission': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'previous_diseases': np.random.poisson(1.5, n_samples),
        'treatment_intensity': np.random.uniform(0, 10, n_samples),
    }

    df = pd.DataFrame(data)

    df['age'] = df['age'].clip(18,100)
    df['blood_pressure'] = df['blood_pressure'].clip(80, 200)
    df['heart_rate'] = df['heart_rate'].clip(50,120)
    df['temperature'] = df['temperature'].clip(35.0, 41.0)
    df['oxygen_saturation'] = df['oxygen_saturation'].clip(90,100)
    df['previous_diseases'] = df['previous_diseases'].clip(0,5)

    base_stay = (
        2 +
        df['age'] * 0.5 +
        df['previous_diseases'] * 1.5 +
        (100 - df['oxygen_saturation']) * 0.8 +
        np.abs(df['blood_pressure'] - 120) * 0.03 +
        np.abs(df['heart_rate'] - 75) * 0.05 +
        df['emergency_admission'] * 2 +
        df['treatment_intensity'] * 0.7
    )

    noise = np.random.normal(0,2, n_samples)
    df['length_of_state'] = np.round(base_stay + noise).clip(1,30)

    df['long_stay'] = (df['length_of_state'] > 5).astype(int)

    return df

df = generate_synthetic_data(1000)

df.to_csv('synthetic_hospital_data.csv', index = False)
print("Синтетические данные сохранены в 'synthetic_hospital_data.csv'")
print(f"Размер датасета: {df.shape}")
print("\nПервые 5 строк:")
print(df.head())