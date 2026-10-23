import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
import matplotlib.pyplot as plt

# Figures are saved here (presentation-ready PNGs) in addition to the usual plt.show()
PLOTS_DIR = "./plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Setup (same dataset/split/target used by every *_optimal.py script)
df = pd.read_csv("./data/reduced_features_v1.csv")

target_col = 'Seebeck Coefficient'

cols_to_drop = ['Composition', 'Site_X', 'Site_Y', 'Site_Z',
                'Seebeck Coefficient']

X = df.drop(columns=cols_to_drop)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Columns worth printing alongside each high-error point. Composition identifies
# the material; the sites show where any unusual element sits; Temperature and
# Electrical Resistivity are the other raw (non-engineered) measured quantities
# that could plausibly explain an error pattern.
context_cols = ['Composition', 'Site_X', 'Site_Y', 'Site_Z', 'Temperature', 'Electrical Resistivity']


def build_results(predictions):
    """Build a results table joined back to composition/metadata for the whole test set."""
    residuals = y_test.values - predictions
    results = df.loc[X_test.index, context_cols].copy()
    results['true_seebeck'] = y_test.values
    results['predicted_seebeck'] = predictions
    results['residual'] = residuals
    results['abs_error'] = np.abs(residuals)
    return results


def top_errors(model_name, results, n=10):
    """Print the n worst rows (by abs_error) with their composition/metadata."""
    worst = results.sort_values('abs_error', ascending=False).head(n)
    print("\n" + "=" * 70)
    print(f"{model_name}: Top {n} highest-error points")
    print("=" * 70)
    print(worst.to_string(index=True))
    return worst


def decile_analysis(model_name, results):
    """Check whether error is concentrated at the extremes of |Seebeck| and
    Temperature, across the WHOLE test set. Binned (decile) means are used
    instead of a plain correlation coefficient because "error is worse at
    extremes" is naturally a U-shape (high at both tails, low in the middle)
    -- a Pearson correlation on the raw signed value would wash that out even
    if the effect is real, since the two tails pull in opposite directions.
    Returns the per-decile mean abs_error AND the actual bin-range labels
    (e.g. "7.3-16.6") so a plot can show real ranges instead of plain decile
    numbers. Binning is done on the true Seebeck/Temperature values, which
    are identical across all four models (only the predictions differ), so
    the bin edges -- and therefore these labels -- are the same regardless
    of which model this is called for."""
    print(f"\n{model_name}: mean abs_error by |Seebeck| decile (extremity, sign-independent)")
    seebeck_bins = pd.qcut(results['true_seebeck'].abs(), q=10, duplicates='drop')
    by_seebeck = results.groupby(seebeck_bins, observed=True)['abs_error'].agg(['mean', 'count'])
    print(by_seebeck.to_string())

    print(f"\n{model_name}: mean abs_error by Temperature decile")
    temp_bins = pd.qcut(results['Temperature'], q=10, duplicates='drop')
    by_temp = results.groupby(temp_bins, observed=True)['abs_error'].agg(['mean', 'count'])
    print(by_temp.to_string())

    # max(0, ...) avoids a "-0" label: qcut's leftmost edge sits a hair below
    # the true minimum (here, -0.0009 instead of 0) due to how it pads bin edges.
    seebeck_labels = [f"{max(0, interval.left):.0f}-{interval.right:.0f}" for interval in by_seebeck.index]
    temp_labels = [f"{interval.left:.0f}-{interval.right:.0f}" for interval in by_temp.index]

    # Raw bin edges, for reuse when counting how many TRAINING rows fall in
    # each same range (see training_density_by_range below).
    seebeck_edges = [by_seebeck.index[0].left] + [interval.right for interval in by_seebeck.index]
    temp_edges = [by_temp.index[0].left] + [interval.right for interval in by_temp.index]

    seebeck_means = by_seebeck['mean'].reset_index(drop=True)
    temp_means = by_temp['mean'].reset_index(drop=True)
    return seebeck_means, temp_means, seebeck_labels, temp_labels, seebeck_edges, temp_edges


def training_density_by_range(seebeck_edges, temp_edges, seebeck_labels, temp_labels):
    """Count how many TRAINING rows fall into each of the same value ranges
    used for the test-set decile analysis above. The test-set 'count' column
    printed by decile_analysis is roughly constant (~61 per bin) by
    construction -- qcut forces equal-SIZED test buckets, so it can't reveal
    whether a range is sparsely represented. This checks the thing the model
    actually learned from: how much TRAINING data supports each range."""
    train_seebeck_counts = pd.cut(y_train.abs(), bins=seebeck_edges, include_lowest=True).value_counts().sort_index()
    train_temp_counts = pd.cut(X_train['Temperature'], bins=temp_edges, include_lowest=True).value_counts().sort_index()

    print("\n" + "=" * 70)
    print("Training-set sample count by |Seebeck| range (same ranges as the decile plot)")
    print("=" * 70)
    print(pd.DataFrame({'range': seebeck_labels, 'train_count': train_seebeck_counts.values}).to_string(index=False))

    print("\n" + "=" * 70)
    print("Training-set sample count by Temperature range (same ranges as the decile plot)")
    print("=" * 70)
    print(pd.DataFrame({'range': temp_labels, 'train_count': train_temp_counts.values}).to_string(index=False))

    return train_seebeck_counts.values, train_temp_counts.values


def cross_model_agreement(worst_tables):
    """Find points that appear in more than one model's top-10 worst-error
    list. Agreement across models with very different inductive biases
    (linear, tree ensembles, kernel) is much stronger evidence of a genuine
    data/material issue than a point only one model gets wrong, which is more
    likely that model's own weakness rather than something wrong with the data."""
    tagged = []
    for model_name, worst in worst_tables.items():
        tmp = worst[['Composition', 'Temperature', 'true_seebeck', 'abs_error']].copy()
        tmp['model'] = model_name
        tagged.append(tmp)
    combined = pd.concat(tagged)

    agreement = combined.groupby(combined.index).agg(
        Composition=('Composition', 'first'),
        Temperature=('Temperature', 'first'),
        true_seebeck=('true_seebeck', 'first'),
        n_models=('model', 'nunique'),
        models=('model', lambda s: ', '.join(sorted(s))),
        mean_abs_error=('abs_error', 'mean'),
    ).sort_values(['n_models', 'mean_abs_error'], ascending=[False, False])

    print("\n" + "=" * 70)
    print("Points appearing in MORE THAN ONE model's Top 10 worst-error list")
    print("=" * 70)
    print(agreement[agreement['n_models'] > 1].to_string(index=True))
    return agreement


def plot_top_errors_grid(worst_tables):
    """One figure, one horizontal-bar panel per model, showing each model's
    top-10 highest-error points labeled by composition."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Top 10 Highest-Error Points by Model', fontsize=16)

    for ax, (model_name, worst) in zip(axes.flatten(), worst_tables.items()):
        ordered = worst.iloc[::-1]  # largest error at the top of the barh
        # Use explicit numeric y-positions rather than the Composition string
        # directly -- barh treats a repeated string as ONE categorical row, so
        # the same composition measured at several temperatures (common in
        # this data) would silently collapse onto a single bar otherwise.
        y_pos = np.arange(len(ordered))
        labels = [f"{comp} ({temp:.0f}K)" for comp, temp in zip(ordered['Composition'], ordered['Temperature'])]
        ax.barh(y_pos, ordered['abs_error'], color='coral')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Absolute Error (μV/K)')
        ax.set_title(model_name)
        ax.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/top_errors_by_model.png", dpi=150)
    plt.show()


def plot_decile_comparison(decile_data, bin_labels, title, xlabel, filename):
    """One line per model, decile bin range (e.g. "7-17") on the x-axis, mean
    abs_error on the y-axis -- lets you compare the shape of the trend (e.g.
    monotonic vs. hump-shaped) across all four models on one chart."""
    plt.figure(figsize=(10, 6))
    for model_name, means in decile_data.items():
        plt.plot(range(len(means)), means.values, marker='o', label=model_name)

    plt.xlabel(xlabel)
    plt.ylabel('Mean Absolute Error (μV/K)')
    plt.title(title)
    plt.xticks(range(len(bin_labels)), bin_labels, rotation=45, ha='right')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{filename}", dpi=150)
    plt.show()


def plot_cross_model_agreement(agreement, filename):
    """Horizontal bars for every point flagged in more than one model's top
    10, colored by HOW MANY models agree on it -- the more models, the
    stronger the evidence this is a real data/material issue rather than one
    model's quirk."""
    shared = agreement[agreement['n_models'] > 1].sort_values(
        ['n_models', 'mean_abs_error'], ascending=[True, True]
    )
    labels = [f"{comp} ({temp:.0f}K)" for comp, temp in zip(shared['Composition'], shared['Temperature'])]
    colors = {2: 'tab:orange', 3: 'tab:red', 4: 'darkred'}
    bar_colors = [colors.get(n, 'gray') for n in shared['n_models']]

    plt.figure(figsize=(10, 7))
    y_pos = np.arange(len(shared))
    plt.barh(y_pos, shared['mean_abs_error'], color=bar_colors)
    plt.yticks(y_pos, labels)
    plt.xlabel('Mean Absolute Error Across Agreeing Models (μV/K)')
    plt.title('Points Flagged as High-Error by Multiple Models')

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    plt.legend(handles, [f"{n} models agree" for n in colors.keys()], loc='lower right')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{filename}", dpi=150)
    plt.show()


def plot_training_density(seebeck_labels, temp_labels, seebeck_train_counts, temp_train_counts,
                           seebeck_deciles, temp_deciles, filename):
    """Two panels (Seebeck, Temperature): training-set sample count per range
    as bars (left axis) against each model's mean abs_error as lines (right
    axis), on the same range bins. Visually shows whether error tracks with
    how much training data supports that range -- if the bars and lines don't
    move together, sparsity isn't the explanation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Training Data Density vs. Model Error, by Range', fontsize=16)

    for ax, labels, train_counts, deciles, xlabel in [
        (ax1, seebeck_labels, seebeck_train_counts, seebeck_deciles, '|Seebeck Coefficient| Range (μV/K)'),
        (ax2, temp_labels, temp_train_counts, temp_deciles, 'Temperature Range (K)'),
    ]:
        x = np.arange(len(labels))
        ax.bar(x, train_counts, color='lightgray', label='Training sample count')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Training Sample Count')

        ax_err = ax.twinx()
        for model_name, means in deciles.items():
            ax_err.plot(x, means.values, marker='o', label=model_name)
        ax_err.set_ylabel('Mean Absolute Error (μV/K)')

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_err.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper center', fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/{filename}", dpi=150)
    plt.show()


# ---- Random Forest ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf = RandomForestRegressor(
    n_estimators=248, max_depth=18, min_samples_split=4, min_samples_leaf=1,
    max_features=0.8022320779230481, max_samples=0.9961419375617291,
    bootstrap=True, random_state=42, n_jobs=-1
)
rf.fit(X_train_scaled, y_train)
rf_results = build_results(rf.predict(X_test_scaled))
rf_worst = top_errors("Random Forest", rf_results)
rf_seebeck_deciles, rf_temp_deciles, seebeck_labels, temp_labels, seebeck_edges, temp_edges = decile_analysis("Random Forest", rf_results)

# ---- XGBoost ----
xgb_reg = xgb.XGBRegressor(
    objective='reg:squarederror', n_estimators=201, max_depth=10,
    learning_rate=0.02588887645789802, gamma=0.7556741452432281, min_child_weight=1,
    subsample=0.7506457892353051, colsample_bytree=0.7601989025296874,
    random_state=42, n_jobs=-1
)
xgb_reg.fit(X_train_scaled, y_train)
xgb_results = build_results(xgb_reg.predict(X_test_scaled))
xgb_worst = top_errors("XGBoost", xgb_results)
xgb_seebeck_deciles, xgb_temp_deciles, _, _, _, _ = decile_analysis("XGBoost", xgb_results)

# ---- SVR ----
svr = SVR(
    kernel='rbf', C=98.49492058649535, epsilon=0.6498734863562683,
    gamma=0.8335370678166605, max_iter=50000
)
svr.fit(X_train_scaled, y_train)
svr_results = build_results(svr.predict(X_test_scaled))
svr_worst = top_errors("SVR", svr_results)
svr_seebeck_deciles, svr_temp_deciles, _, _, _, _ = decile_analysis("SVR", svr_results)

# ---- Lasso (own preprocessing: clip leverage outliers, then RobustScaler) ----
lasso_X_train = X_train.copy()
lasso_X_test = X_test.copy()
clip_lower = lasso_X_train.quantile(0.01)
clip_upper = lasso_X_train.quantile(0.99)
lasso_X_train = lasso_X_train.clip(lower=clip_lower, upper=clip_upper, axis=1)
lasso_X_test = lasso_X_test.clip(lower=clip_lower, upper=clip_upper, axis=1)

lasso_scaler = RobustScaler()
lasso_X_train_scaled = lasso_scaler.fit_transform(lasso_X_train)
lasso_X_test_scaled = lasso_scaler.transform(lasso_X_test)

lasso = Lasso(alpha=0.0060452910025173895, selection='random', max_iter=10000, random_state=42)
lasso.fit(lasso_X_train_scaled, y_train)
lasso_results = build_results(lasso.predict(lasso_X_test_scaled))
lasso_worst = top_errors("Lasso", lasso_results)
lasso_seebeck_deciles, lasso_temp_deciles, _, _, _, _ = decile_analysis("Lasso", lasso_results)

# ---- Cross-model agreement and training density ----
worst_tables = {
    "Random Forest": rf_worst,
    "XGBoost": xgb_worst,
    "SVR": svr_worst,
    "Lasso": lasso_worst,
}
agreement = cross_model_agreement(worst_tables)
seebeck_train_counts, temp_train_counts = training_density_by_range(
    seebeck_edges, temp_edges, seebeck_labels, temp_labels
)

# ---- Plots for the presentation ----
plot_top_errors_grid(worst_tables)

seebeck_deciles = {
    "Random Forest": rf_seebeck_deciles,
    "XGBoost": xgb_seebeck_deciles,
    "SVR": svr_seebeck_deciles,
    "Lasso": lasso_seebeck_deciles,
}
plot_decile_comparison(
    seebeck_deciles,
    seebeck_labels,
    "Mean Absolute Error by |Seebeck Coefficient| Decile",
    "|Seebeck Coefficient| Range (μV/K)",
    "decile_seebeck_comparison.png",
)

temp_deciles = {
    "Random Forest": rf_temp_deciles,
    "XGBoost": xgb_temp_deciles,
    "SVR": svr_temp_deciles,
    "Lasso": lasso_temp_deciles,
}
plot_decile_comparison(
    temp_deciles,
    temp_labels,
    "Mean Absolute Error by Temperature Decile",
    "Temperature Range (K)",
    "decile_temperature_comparison.png",
)

plot_cross_model_agreement(agreement, "cross_model_agreement.png")

plot_training_density(
    seebeck_labels, temp_labels, seebeck_train_counts, temp_train_counts,
    seebeck_deciles, temp_deciles, "training_density_vs_error.png",
)
