"""
   Task 4: 
   Movie Rating Prediction """

import os
import warnings
import subprocess
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import joblib
from tqdm import tqdm
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from math import sqrt


# ---------------------------
# Settings 
# ---------------------------
warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.facecolor": "#f7fbff",
    "axes.facecolor": "#f7fbff",
    "savefig.facecolor": "#f7fbff",
    "figure.figsize": (10,6),
    "axes.grid": True,
    "grid.color": "#e6eef8",
    "axes.edgecolor": "#333333",
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
    "legend.frameon": True,
})

PALETTE = [
    "#FF6B6B", "#4ECDC4", "#556B2F", "#FFD166", "#6A4C93",
    "#FFA07A", "#2E8B57", "#00A5CF", "#FFB400", "#A0D468",
    "#FF6F91", "#4D96FF", "#7BD389", "#E76F51", "#9B5DE5",
]

# ---------------------------
# I/O paths
# ---------------------------
BASE_DIR = os.getcwd()
RATING_FILE = r"C:/Users/Abdullah Umer/Desktop/Arch Technologies Internship/Task 4/rating.csv"
MOVIES_FILE = os.path.join(BASE_DIR, "movies.csv")
GENOME_FILE = os.path.join(BASE_DIR, "genome_scores.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Working directory: {BASE_DIR}")
print("Output directory:", OUTPUT_DIR)

def save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("Saved:", path)


def ensure_cols_int(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(pd.Int64Dtype())


# ---------------------------
# Load dataset 
# ---------------------------
if not os.path.exists(RATING_FILE):
    raise FileNotFoundError(f"rating.csv not found in {BASE_DIR}. Please put rating.csv there and re-run.")

print("Loading rating dataset...")
df = pd.read_csv(RATING_FILE, low_memory=False)
df.columns = [c.strip() for c in df.columns]

expected = set(["userId", "movieId", "rating"])
if not expected.issubset(set(df.columns)):
    cols_lower = {c.lower(): c for c in df.columns}
    if 'userid' in cols_lower:
        df.rename(columns={cols_lower['userid']:"userId"}, inplace=True)
    if 'movieid' in cols_lower:
        df.rename(columns={cols_lower['movieid']:"movieId"}, inplace=True)
    if 'rating' not in df.columns and 'ratings' in cols_lower:
        df.rename(columns={cols_lower['ratings']:"rating"}, inplace=True)

if not expected.issubset(set(df.columns)):
    raise ValueError(f"rating.csv must contain columns userId, movieId, rating. Found: {df.columns.tolist()}")

df = df[['userId', 'movieId', 'rating'] + ([c for c in df.columns if c not in ['userId','movieId','rating']] or [])]
df['userId'] = pd.to_numeric(df['userId'], errors='coerce').astype('Int64')
df['movieId'] = pd.to_numeric(df['movieId'], errors='coerce').astype('Int64')
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df = df.dropna(subset=['userId','movieId','rating'])
df['userId'] = df['userId'].astype(int)
df['movieId'] = df['movieId'].astype(int)

print("Rows after dropping nulls:", df.shape[0])
print("Unique users:", df['userId'].nunique())
print("Unique movies:", df['movieId'].nunique())



# ---------------------------
# EDA & feature engineering
# ---------------------------
print("Starting EDA and feature engineering...")

n_users = df['userId'].nunique()
n_movies = df['movieId'].nunique()
n_ratings = len(df)
sparsity = 1 - (n_ratings / (n_users * n_movies))
print(f"Users: {n_users}, Movies: {n_movies}, Ratings: {n_ratings}, Sparsity: {sparsity:.6f}")

user_counts = df.groupby('userId').size().rename("ratings_count_user")
movie_counts = df.groupby('movieId').size().rename("ratings_count_movie")
movie_avg = df.groupby('movieId')['rating'].mean().rename("movie_avg_rating")
user_avg = df.groupby('userId')['rating'].mean().rename("user_avg_rating")

df = df.merge(user_counts, on='userId', how='left')
df = df.merge(movie_counts, on='movieId', how='left')
df = df.merge(movie_avg, on='movieId', how='left')
df = df.merge(user_avg, on='userId', how='left')

# safe cap for counts (for plotting)
df['ratings_count_user_capped'] = df['ratings_count_user'].clip(upper=200)
df['ratings_count_movie_capped'] = df['ratings_count_movie'].clip(upper=200)

# Add user_movie_avg_diff
df['user_movie_diff'] = df['rating'] - df['movie_avg_rating']

# Prepare features for regression model
# We'll create a small set of features: user_avg, movie_avg, user_count, movie_count
features = df[['userId','movieId','rating','user_avg_rating','movie_avg_rating','ratings_count_user','ratings_count_movie']].drop_duplicates()





# ---------------------------
# Visualizations (15)
# Each figure saved to output/
# ---------------------------
print("Creating and saving visualizations...")

# Helper: create histogram
def plot_hist(series, title, fname, bins=30, color=None):
    fig, ax = plt.subplots()
    c = color if color else PALETTE[0]
    ax.hist(series.dropna(), bins=bins, edgecolor='white', linewidth=0.7, alpha=0.95)
    ax.set_title(title)
    ax.set_xlabel(series.name)
    ax.set_ylabel("Count")
    save_fig(fig, fname)

# 1) Rating distribution
plot_hist(df['rating'], "1) Rating Distribution", "01_rating_distribution.png", bins=20)

# 2) Ratings per user (distribution) - log scale visual
fig, ax = plt.subplots()
ax.hist(df['ratings_count_user'], bins=50)
ax.set_title("2) Ratings per User (distribution)")
ax.set_xlabel("Ratings per user")
ax.set_ylabel("Count")
ax.set_yscale('log')
save_fig(fig, "02_ratings_per_user_log.png")

# 3) Ratings per movie (distribution) - log scale
fig, ax = plt.subplots()
ax.hist(df['ratings_count_movie'], bins=50)
ax.set_title("3) Ratings per Movie (distribution)")
ax.set_xlabel("Ratings per movie")
ax.set_ylabel("Count")
ax.set_yscale('log')
save_fig(fig, "03_ratings_per_movie_log.png")

# 4) Top 20 movies by number of ratings (if movies available show titles)


# 5) Top 20 users by rating count
top_users = user_counts.sort_values(ascending=False).head(20)
fig, ax = plt.subplots(figsize=(10,6))
ax.bar(range(len(top_users)), top_users.values, color=PALETTE[:len(top_users)])
ax.set_xticks(range(len(top_users)))
ax.set_xticklabels(top_users.index.astype(str), rotation=45)
ax.set_title("5) Top 20 Users by Rating Count")
save_fig(fig, "05_top20_users_by_ratings.png")

# 6) Boxplot of ratings
fig, ax = plt.subplots()
ax.boxplot(df['rating'].dropna(), vert=False)
ax.set_title("6) Rating Boxplot")
save_fig(fig, "06_rating_boxplot.png")

# 7) Movie average rating vs number of ratings (scatter)
agg = pd.DataFrame({
    'movie_avg': movie_avg,
    'movie_count': movie_counts
}).reset_index()
fig, ax = plt.subplots()
ax.scatter(agg['movie_count'], agg['movie_avg'], alpha=0.6, s=30, c=PALETTE[1])
ax.set_xscale('log')
ax.set_xlabel("Number of ratings (log)")
ax.set_ylabel("Average rating")
ax.set_title("7) Movie Avg Rating vs Rating Count")
save_fig(fig, "07_movie_avg_vs_count.png")

# 8) User average rating vs number of ratings (scatter)
uagg = pd.DataFrame({
    'user_avg': user_avg,
    'user_count': user_counts
}).reset_index()
fig, ax = plt.subplots()
ax.scatter(uagg['user_count'], uagg['user_avg'], alpha=0.6, s=30, c=PALETTE[2])
ax.set_xscale('log')
ax.set_xlabel("Number of ratings by user (log)")
ax.set_ylabel("User average rating")
ax.set_title("8) User Avg Rating vs Rating Count")
save_fig(fig, "08_user_avg_vs_count.png")

# 9) Sparsity heatmap sample (small subset)
# create a pivot for top N users and movies
def plot_sparse_sample(df, top_n_users=50, top_n_movies=50, fname="09_sparsity_sample.png"):
    uc = df.groupby('userId').size().sort_values(ascending=False).head(top_n_users).index
    mc = df.groupby('movieId').size().sort_values(ascending=False).head(top_n_movies).index
    sample = df[df['userId'].isin(uc) & df['movieId'].isin(mc)]
    pivot = sample.pivot_table(index='userId', columns='movieId', values='rating', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(10,7))
    c = ax.imshow(pivot.notna(), aspect='auto', interpolation='none', cmap='Greys')
    ax.set_title("9) Sparsity sample (top users x top movies) (black=rating present)")
    ax.set_xlabel("Movies")
    ax.set_ylabel("Users")
    save_fig(fig, fname)

plot_sparse_sample(df, top_n_users=60, top_n_movies=90, fname="09_sparsity_sample.png")

# 10) Time-based plot (if timestamp exists)
if 'timestamp' in pd.read_csv(RATING_FILE, nrows=2).columns:
    try:
        raw = pd.read_csv(RATING_FILE, usecols=['timestamp'], low_memory=False)
        # try multiple common formats
        def parse_ts(x):
            for fmt in ("%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M", "%d-%b-%y", "%Y-%m-%d"):
                try:
                    return pd.to_datetime(x, format=fmt)
                except Exception:
                    continue
            try:
                return pd.to_datetime(x, errors='coerce')
            except:
                return pd.NaT
        raw['ts_parsed'] = raw['timestamp'].apply(parse_ts)
        raw = raw.dropna(subset=['ts_parsed'])
        if len(raw) > 0:
            raw['year'] = raw['ts_parsed'].dt.year
            counts_by_year = raw.groupby('year').size()
            fig, ax = plt.subplots()
            ax.plot(counts_by_year.index.astype(int), counts_by_year.values, marker='o', color=PALETTE[3])
            ax.set_title("10) Ratings per Year (from timestamp)")
            ax.set_xlabel("Year")
            ax.set_ylabel("Number of ratings")
            save_fig(fig, "10_ratings_per_year.png")
    except Exception as e:
        print("Could not parse timestamps for time-based plot:", e)

# 11) Distribution of user_movie_diff (rating minus movie avg)
plot_hist(df['user_movie_diff'], "11) Distribution: rating - movie_avg_rating", "11_user_movie_diff.png", bins=40)

# 12) Residuals after simple baseline (rating - user_avg)
df['baseline_user_diff'] = df['rating'] - df['user_avg_rating']
plot_hist(df['baseline_user_diff'], "12) Distribution: rating - user_avg_rating (baseline)", "12_baseline_user_diff.png", bins=40)


# 14) Model baseline: Distribution of movie_avg_rating
plot_hist(movie_avg, "14) Distribution of Movie Average Ratings", "14_movie_avg_distribution.png", bins=40)

# 15) Scatter: user_avg vs movie_avg (sample)
sample = pd.merge(user_avg.rename('user_avg').reset_index(), movie_avg.rename('movie_avg').reset_index(), how='cross').sample(5000, random_state=42)
fig, ax = plt.subplots()
ax.scatter(sample['user_avg'], sample['movie_avg'], alpha=0.3, s=10, c=PALETTE[4])
ax.set_title("15) Sample: User Avg vs Movie Avg (scatter)")
ax.set_xlabel("User Avg")
ax.set_ylabel("Movie Avg")
save_fig(fig, "15_user_avg_vs_movie_avg_sample.png")

print("All visualizations completed.")








# ---------------------------
# Collaborative Filtering with Truncated SVD (replacement for Surprise)
# ---------------------------
print("Modeling: Approximate Collaborative Filtering with TruncatedSVD and RandomForest baseline")

# Map userId and movieId to indices for sparse matrix
user_ids = df['userId'].unique()
movie_ids = df['movieId'].unique()
user_to_idx = {u: i for i, u in enumerate(user_ids)}
movie_to_idx = {m: i for i, m in enumerate(movie_ids)}

df['user_idx'] = df['userId'].map(user_to_idx)
df['movie_idx'] = df['movieId'].map(movie_to_idx)

# Train-test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

n_users = len(user_ids)
n_movies = len(movie_ids)

# Create sparse matrix from train data
train_sparse = coo_matrix((train_df['rating'], (train_df['user_idx'], train_df['movie_idx'])), shape=(n_users, n_movies))

# Fit TruncatedSVD on train matrix
n_factors = 50
svd_model = TruncatedSVD(n_components=n_factors, random_state=42)
print("Training TruncatedSVD model...")
svd_model.fit(train_sparse)

U = svd_model.transform(train_sparse)                  # User factors (n_users x n_factors)
Sigma = svd_model.singular_values_                     # Singular values (n_factors,)
VT = svd_model.components_                             # Movie factors (n_factors x n_movies)

sigma_diag = np.diag(Sigma)
user_factors = U                                        # Shape: n_users x n_factors
movie_factors = VT.T                                   # Shape: n_movies x n_factors

# Reconstruct approx rating matrix
reconstructed = user_factors @ sigma_diag @ movie_factors.T

# Predict on test set
test_users_idx = test_df['user_idx'].values
test_movies_idx = test_df['movie_idx'].values
true_ratings = test_df['rating'].values

predicted_ratings = reconstructed[test_users_idx, test_movies_idx]

# Clip predictions to rating scale
min_rating = df['rating'].min()
max_rating = df['rating'].max()
predicted_ratings = np.clip(predicted_ratings, min_rating, max_rating)

mse = mean_squared_error(true_ratings, predicted_ratings)
svd_rmse = sqrt(mse)
svd_mae = mean_absolute_error(true_ratings, predicted_ratings)
print(f"Approximate SVD -> RMSE: {svd_rmse:.4f}, MAE: {svd_mae:.4f}")



# Save SVD factors and mappings
joblib.dump({
    'user_factors': user_factors,
    'sigma': Sigma,
    'movie_factors': movie_factors,
    'user_to_idx': user_to_idx,
    'movie_to_idx': movie_to_idx
}, os.path.join(OUTPUT_DIR, "approx_svd_model.joblib"))
print("Saved approximate SVD model (factors and mappings)")

# ---------------------------
# Random Forest baseline
# ---------------------------
print("Preparing features for RandomForest regression...")

reg_df = df[['userId','movieId','rating','user_avg_rating','movie_avg_rating','ratings_count_user','ratings_count_movie']].drop_duplicates()

uid_enc = LabelEncoder()
mid_enc = LabelEncoder()
reg_df['user_enc'] = uid_enc.fit_transform(reg_df['userId'])
reg_df['movie_enc'] = mid_enc.fit_transform(reg_df['movieId'])

X = reg_df[['user_avg_rating','movie_avg_rating','ratings_count_user','ratings_count_movie','user_enc','movie_enc']].fillna(0)
y = reg_df['rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
print("Training RandomForest model...")
rf.fit(X_train, y_train)
preds = rf.predict(X_test)

mse_rf = mean_squared_error(y_test, preds)
rf_rmse = sqrt(mse_rf)
rf_mae = mean_absolute_error(y_test, preds)
print(f"RandomForest -> RMSE: {rf_rmse:.4f}, MAE: {rf_mae:.4f}")

joblib.dump(rf, os.path.join(OUTPUT_DIR, "rf_model.joblib"))
print("Saved RandomForest model")



# ---------------------------
# Plot comparison of models
# ---------------------------
fig, ax = plt.subplots()
models = ['Approx SVD', 'RandomForest']
rmses = [svd_rmse, rf_rmse]
maes = [svd_mae, rf_mae]
x = np.arange(len(models))
width = 0.35
ax.bar(x - width/2, rmses, width, label='RMSE', color=PALETTE[0])
ax.bar(x + width/2, maes, width, label='MAE', color=PALETTE[1])
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("Error")
ax.set_title("Model comparison (RMSE & MAE)")
ax.legend()
save_fig(fig, "model_comparison_rmse_mae.png")



# ---------------------------
# Feature importance plot for RF
try:
    importances = rf.feature_importances_
    feat_names = X.columns.tolist()
    idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.barh(range(len(importances)), importances[idx], color=PALETTE[:len(importances)])
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels([feat_names[i] for i in idx])
    ax.set_title("RandomForest Feature Importances")
    save_fig(fig, "rf_feature_importances.png")
except Exception as e:
    print("Could not plot feature importances:", e)




# ---------------------------
# Summary
report = {
    "n_users": int(n_users),
    "n_movies": int(n_movies),
    "n_ratings": int(n_ratings),
    "sparsity": float(sparsity),
    "svd_rmse": float(svd_rmse),
    "svd_mae": float(svd_mae),
    "rf_rmse": float(rf_rmse),
    "rf_mae": float(rf_mae),
    "generated_plots": [f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith(".png")]
}

import json
with open(os.path.join(OUTPUT_DIR, "summary_report.json"), "w") as f:
    json.dump(report, f, indent=2)
print("Saved summary_report.json in output/")

print("\n--- Completed pipeline ---")
print(f"Check the '{OUTPUT_DIR}' folder for visualizations, models, and summary files.")





