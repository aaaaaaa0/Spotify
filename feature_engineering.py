# -*- coding: utf-8 -*-
"""
Feature Engineering для Spotify Dataset
Оптимизировано для больших датасетов (~114k треков)
Улучшенная версия с обработкой ошибок и расширенной аналитикой
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df):
    """Создание новых признаков с защитой от бесконечных значений"""
    df_fe = df.copy()
    
    print("🔄 Создание новых признаков...")
    
    # Базовые преобразования
    df_fe['duration_min'] = df_fe['duration_ms'] / 60000
    df_fe['explicit_int'] = df_fe['explicit'].astype(int)
    
    # Взаимодействие признаков
    df_fe['energy_danceability'] = df_fe['energy'] * df_fe['danceability']
    df_fe['acoustic_energy_ratio'] = df_fe['acousticness'] / (df_fe['energy'] + 1e-6)
    df_fe['speech_to_acoustic'] = df_fe['speechiness'] / (df_fe['acousticness'] + 1e-6)
    df_fe['mood_index'] = (df_fe['valence'] + df_fe['energy']) / 2
    
    # Сложные характеристики
    df_fe['complexity_index'] = (
        df_fe['speechiness'] + df_fe['instrumentalness'] + df_fe['liveness']
    ) / 3
    
    # Нормализация loudness
    df_fe['loudness_normalized'] = (
        (df_fe['loudness'] - df_fe['loudness'].min()) /
        (df_fe['loudness'].max() - df_fe['loudness'].min())
    )
    
    # Защита от inf/-inf в признаках с делением
    ratio_features = ['acoustic_energy_ratio', 'speech_to_acoustic']
    for feature in ratio_features:
        df_fe[feature] = df_fe[feature].replace([np.inf, -np.inf], 0)
        df_fe[feature] = df_fe[feature].fillna(0)
    
    print(f"✅ Создано {len(df_fe.columns) - len(df.columns)} новых признаков")
    return df_fe

# ============================================================
# CORRELATION ANALYSIS
# ============================================================

def get_feature_correlations(df_fe, df_original, target_col='popularity'):
    """Анализ корреляции новых признаков с улучшенной визуализацией"""
    original_cols = set(df_original.columns)
    new_features = list(set(df_fe.columns) - original_cols)

    if not new_features:
        print("⚠️ Нет новых признаков для анализа")
        return None, []

    if target_col not in df_fe.columns:
        print(f"❌ Ошибка: {target_col} отсутствует в данных")
        return None, new_features

    try:
        correlations = df_fe[new_features + [target_col]].corr()[target_col].sort_values(ascending=False)
        
        print("\n" + "="*60)
        print("📊 КОРРЕЛЯЦИЯ НОВЫХ ПРИЗНАКОВ С ПОПУЛЯРНОСТЬЮ")
        print("="*60)
        
        # Разделяем положительные и отрицательные корреляции
        pos_corr = correlations[correlations > 0].drop(target_col, errors='ignore')
        neg_corr = correlations[correlations < 0].drop(target_col, errors='ignore')
        
        if not pos_corr.empty:
            print("\n📈 Положительная корреляция (топ-5):")
            for feature, corr in pos_corr.head(5).items():
                print(f"   {feature:<30} {corr:+.3f}")
        
        if not neg_corr.empty:
            print("\n📉 Отрицательная корреляция (топ-5):")
            for feature, corr in neg_corr.head(5).items():
                print(f"   {feature:<30} {corr:.3f}")
                
        # Самые значимые по абсолютному значению
        abs_corr = correlations.drop(target_col, errors='ignore').abs().sort_values(ascending=False)
        print(f"\n🎯 Самые значимые признаки (|corr| > 0.1):")
        significant_features = abs_corr[abs_corr > 0.1]
        for feature, corr in significant_features.items():
            original_corr = correlations[feature]
            print(f"   {feature:<30} {original_corr:+.3f}")
            
    except Exception as e:
        print(f"❌ Ошибка при расчете корреляций: {e}")
        return None, new_features

    return correlations, new_features

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def calculate_feature_importance(df_fe, target_col='popularity', numeric_features=None):
    """Расчёт важности признаков с помощью ExtraTrees"""
    if numeric_features is None:
        numeric_features = [col for col in df_fe.select_dtypes(include=[np.number]).columns
                            if col != target_col]
    
    # Заполнение пропущенных значений
    X = df_fe[numeric_features].fillna(0)
    y = df_fe[target_col]
    
    print(f"🔍 Расчет важности признаков для {len(numeric_features)} фич...")
    
    rf = ExtraTreesRegressor(
        n_estimators=80,
        max_depth=12,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X, y)
    
    feature_importances = pd.DataFrame({
        'feature': numeric_features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🏆 Топ-15 самых важных признаков:")
    print("-" * 50)
    for i, (_, row) in enumerate(feature_importances.head(15).iterrows(), 1):
        print(f"   {i:2d}. {row['feature']:<25} {row['importance']:.4f}")

    return feature_importances, rf

# ============================================================
# SELECT BEST FEATURES
# ============================================================

def get_final_feature_set(df_fe, top_n=15, target_col='popularity'):
    """Выбор топ-N признаков для моделирования с валидацией"""
    if target_col not in df_fe.columns:
        raise ValueError(f"❌ Целевая переменная '{target_col}' не найдена в данных")
    
    numeric_features = [col for col in df_fe.select_dtypes(include=[np.number]).columns
                        if col != target_col]
    
    if len(numeric_features) == 0:
        raise ValueError("❌ Не найдено числовых признаков для анализа")

    feature_importances, _ = calculate_feature_importance(
        df_fe,
        target_col=target_col,
        numeric_features=numeric_features
    )

    selected_features = feature_importances.head(top_n)['feature'].tolist()
    
    print(f"\n🎯 Выбрано {len(selected_features)} признаков для модели:")
    print("-" * 50)
    for i, feature in enumerate(selected_features, 1):
        importance = feature_importances.loc[feature_importances['feature'] == feature, 'importance'].iloc[0]
        print(f"   {i:2d}. {feature:<25} {importance:.4f}")
    
    return selected_features

# ============================================================
# FEATURE QUALITY ANALYSIS
# ============================================================

def analyze_feature_quality(df_fe, selected_features, target_col='popularity'):
    """Анализ качества отобранных признаков"""
    print("\n" + "="*60)
    print("🔍 АНАЛИЗ КАЧЕСТВА ПРИЗНАКОВ")
    print("="*60)
    
    X = df_fe[selected_features]
    
    # Проверка на мультиколлинеарность
    corr_matrix = X.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = []
    
    for col1 in upper_triangle.columns:
        for col2 in upper_triangle.columns:
            if upper_triangle.loc[col1, col2] > 0.8:
                high_corr_pairs.append((col1, col2, upper_triangle.loc[col1, col2]))
    
    if high_corr_pairs:
        print("⚠️  Высокая корреляция между признаками (> 0.8):")
        for col1, col2, corr in high_corr_pairs[:5]:  # Показываем первые 5
            print(f"   {col1:<25} ↔ {col2:<25} {corr:.3f}")
    else:
        print("✅ Мультиколлинеарность в норме (все корреляции < 0.8)")
    
    # Дисперсия признаков
    variances = X.var().sort_values(ascending=False)
    low_variance = variances[variances < 0.01]
    
    print(f"\n📊 Дисперсия признаков:")
    print(f"   Средняя дисперсия: {variances.mean():.4f}")
    print(f"   Признаков с низкой дисперсией (< 0.01): {len(low_variance)}")
    
    if not low_variance.empty:
        print("   Низкодисперсные признаки:")
        for feature, variance in low_variance.items():
            print(f"     {feature}: {variance:.6f}")

# ============================================================
# PREPARE DATA FOR MODEL
# ============================================================

def prepare_features_for_modeling(df_fe, features_list, target_col='popularity'):
    """Подготовка X и y для модели с валидацией"""
    missing_features = [f for f in features_list if f not in df_fe.columns]
    if missing_features:
        raise ValueError(f"❌ Отсутствуют признаки: {missing_features}")
    
    X = df_fe[features_list].fillna(0)
    y = df_fe[target_col]
    
    print(f"✅ Данные подготовлены: X{X.shape}, y{y.shape}")
    return X, y

# ============================================================
# MAIN EXECUTION (пример использования)
# ============================================================

if __name__ == "__main__":
    try:
        print("🚀 ЗАПУСК FEATURE ENGINEERING")
        print("=" * 50)
        
        # Загрузка данных
        df = pd.read_csv('dataset.csv')
        print(f"✅ Данные загружены: {df.shape}")
        
        # Создание новых признаков
        df_fe = create_features(df)
        print(f"✅ Признаки созданы: +{len(df_fe.columns) - len(df.columns)} новых")
        
        # Анализ корреляций новых признаков
        correlations, new_features = get_feature_correlations(df_fe, df)
        
        # Выбор лучших признаков
        selected_features = get_final_feature_set(df_fe, top_n=15)
        
        # Анализ качества признаков
        analyze_feature_quality(df_fe, selected_features)
        
        # Подготовка данных для модели
        X, y = prepare_features_for_modeling(df_fe, selected_features)
        print(f"✅ Данные для модели готовы: X{X.shape}, y{y.shape}")
        
        print("\n🎉 FEATURE ENGINEERING УСПЕШНО ЗАВЕРШЁН!")
        print("=" * 50)
        
    except FileNotFoundError:
        print("❌ Ошибка: файл dataset.csv не найден")
        print("   Убедитесь, что файл находится в правильной директории")
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        