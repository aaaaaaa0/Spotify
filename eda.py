import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Matplotlib / Seaborn settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16

filepath = r'C:\Users\717ll\OneDrive\Documents\3 семестр\23. АДИИ\Spotify\dataset.csv'


# =====================================================================
# DATA LOAD
# =====================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset and display basic info."""
    df = pd.read_csv(filepath, index_col=0)

    print(f"Размер датасета: {df.shape}\n")
    print("Первые 3 строки:")
    print(df.head(3), "\n")

    print("Информация о данных:")
    df.info()
    return df


# =====================================================================
# MISSING VALUES
# =====================================================================

def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Find and print missing values."""
    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)

    missing_df = (
        pd.DataFrame({
            'missing_count': missing,
            'missing_percentage': missing_pct
        })
        .query("missing_count > 0")
        .sort_values("missing_count", ascending=False)
    )

    if missing_df.empty:
        print("Пропущенных значений НЕТ")
    else:
        print("Пропущенные значения:")
        print(missing_df)

    return missing_df


# =====================================================================
# POPULARITY DISTRIBUTION
# =====================================================================

def plot_target_distribution(df: pd.DataFrame, target_col: str = 'popularity'):
    """Plot histogram + boxplot of popularity."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.histplot(df[target_col], bins=50, kde=True,
                 ax=axes[0], color='#1DB954')
    axes[0].set_title('Распределение популярности треков')
    axes[0].set_xlabel('Popularity')

    sns.boxplot(y=df[target_col], ax=axes[1],
                color='#1DB954', width=0.4)
    axes[1].set_title('Boxplot популярности')

    plt.tight_layout()
    plt.show()

    print("\nСтатистики популярности:")
    print(df[target_col].describe().round(2))
    print(f"Skewness: {df[target_col].skew():.3f}")
    print(f"Kurtosis: {df[target_col].kurtosis():.3f}")


# =====================================================================
# CORRELATIONS
# =====================================================================

def plot_correlations(df: pd.DataFrame, target_col='popularity'):
    """Plot numeric correlations with popularity."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()[target_col].drop(target_col).sort_values()

    plt.figure(figsize=(10, 8))

    # FIX: provide hue to avoid FutureWarning
    sns.barplot(
        x=corr.values,
        y=corr.index,
        hue=corr.index,
        dodge=False,
        legend=False,
        palette="coolwarm"
    )

    plt.title('Корреляция признаков с popularity')
    plt.xlabel('Корреляция')
    plt.axvline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.show()

    print("\nТоп-10 признаков по |корреляции| с popularity:")
    print(corr.abs().sort_values(ascending=False).head(10).round(4))

    return corr


# =====================================================================
# GENRE POPULARITY
# =====================================================================

def plot_genre_popularity(df: pd.DataFrame, top_n=20):
    """Plot average popularity for each genre."""
    genre_pop = df.groupby('track_genre')['popularity'].mean().sort_values(ascending=False)
    top_genres = genre_pop.head(top_n)

    plt.figure(figsize=(12, 8))

    sns.barplot(
        x=top_genres.values,
        y=top_genres.index,
        hue=top_genres.index,
        dodge=False,
        legend=False,
        palette="viridis"
    )

    plt.title(f'TOP-{top_n} популярных жанров')
    plt.xlabel('Средняя популярность')
    plt.tight_layout()
    plt.show()

    print(f"\nСамый популярный жанр: {genre_pop.idxmax()} — {genre_pop.max():.1f}")
    print(f"Самый непопулярный: {genre_pop.idxmin()} — {genre_pop.min():.1f}")


# =====================================================================
# EXPLICIT EFFECT
# =====================================================================

def plot_explicit_effect(df: pd.DataFrame):
    """Plot explicit vs popularity."""
    explicit_map = df['explicit'].map({False: 'Нет', True: 'Да'})
    means = df.groupby('explicit')['popularity'].mean()

    plt.figure(figsize=(9, 6))

    sns.boxplot(
        x=explicit_map,
        y=df['popularity'],
        hue=explicit_map,
        legend=False,
        palette={'Нет': '#8c8c8c', 'Да': '#1DB954'}
    )

    plt.title('Влияние explicit на популярность')
    plt.xlabel('Explicit контент')
    plt.ylabel('Popularity')

    plt.text(0, means[False] + 1, f'{means[False]:.1f}', ha='center', fontweight='bold')
    plt.text(1, means[True] + 1, f'{means[True]:.1f}', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.show()

    print("\nСредняя популярность по explicit:")
    print(means.round(2))


# =====================================================================
# RUN EDA
# =====================================================================

df = load_data(filepath)

print("\n" + "=" * 60)
analyze_missing_values(df)

print("\n" + "=" * 60)
plot_target_distribution(df)

print("\n" + "=" * 60)
corr = plot_correlations(df)

print("\n" + "=" * 60)
plot_genre_popularity(df, top_n=20)

print("\n" + "=" * 60)
plot_explicit_effect(df)

print("\nEDA успешно завершён!")
