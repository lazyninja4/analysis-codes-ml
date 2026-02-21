import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("microbial diversity analysis")
print("multiple metrics, distance-based")
print("=" * 70)

print("\nstep 1: loading microbiome data")
samples = pd.read_csv('microbiome_data.csv')

base_cols = ['sample_id', 'latitude', 'longitude']
if not all(c in samples.columns for c in base_cols):
    raise ValueError(f"missing required columns: {base_cols}")

print(f"loaded {len(samples)} samples")

print("\nstep 2: setting up diversity metrics")

if 'shannon_diversity' in samples.columns:
    print("✓ shannon diversity found")
else:
    print("shannon diversity missing")

if 'species_richness' in samples.columns or 'observed_otus' in samples.columns:
    richness_col = 'species_richness' if 'species_richness' in samples.columns else 'observed_otus'
    if 'shannon_diversity' in samples.columns:
        samples['evenness'] = samples['shannon_diversity'] / np.log(samples[richness_col])
        print("✓ evenness calculated")

        if 'simpson' in samples.columns:
            samples['inv_simpson'] = 1 / samples['simpson']
            print("✓ inverse simpson calculated")
            metrics = ['shannon_diversity', 'evenness', 'inv_simpson']
        else:
            metrics = ['shannon_diversity', 'evenness']
    else:
        metrics = ['shannon_diversity'] if 'shannon_diversity' in samples.columns else []
else:
    metrics = ['shannon_diversity'] if 'shannon_diversity' in samples.columns else []

if not metrics:
    raise ValueError("no usable diversity metrics")

print(f"using metrics: {metrics}")

print("\nstep 3: loading pollution sources")
sources = pd.read_csv('pollution_sources.csv')

src_cols = ['source_name', 'latitude', 'longitude']
if not all(c in sources.columns for c in src_cols):
    raise ValueError(f"pollution file missing columns: {src_cols}")

print(f"loaded {len(sources)} pollution sources")

print("\nfiltering pollution data")
sources_filt = sources.copy()

before = len(sources_filt)
sources_filt = sources_filt.drop_duplicates(subset=['latitude', 'longitude'])
print(f"removed {before - len(sources_filt)} duplicate locations")

lat_min, lat_max = samples['latitude'].min() - 2, samples['latitude'].max() + 2
lon_min, lon_max = samples['longitude'].min() - 2, samples['longitude'].max() + 2

sources_filt = sources_filt[
    (sources_filt['latitude'] >= lat_min) &
    (sources_filt['latitude'] <= lat_max) &
    (sources_filt['longitude'] >= lon_min) &
    (sources_filt['longitude'] <= lon_max)
]

print(f"sources after region filter: {len(sources_filt)}")

def spatial_filter(df, min_km=5):
    coords = df[['latitude', 'longitude']].values
    keep = []
    for i in range(len(coords)):
        if i in keep:
            continue
        keep.append(i)
        for j in range(i + 1, len(coords)):
            if j not in keep:
                if geodesic(coords[i], coords[j]).km < min_km:
                    pass
    return df.iloc[keep]

before = len(sources_filt)
sources_filt = spatial_filter(sources_filt, min_km=10)
print(f"sources after spacing filter: {len(sources_filt)}")
print(f"total filtering: {len(sources)} → {len(sources_filt)}")

print("\nstep 4: computing distances")

def nearest_source_dist(sample_xy, source_xy):
    return min(geodesic(sample_xy, s).km for s in source_xy)

source_xy = sources_filt[['latitude', 'longitude']].values

dist_km = []
for _, row in samples.iterrows():
    dist_km.append(nearest_source_dist((row['latitude'], row['longitude']), source_xy))

samples['distance_km'] = dist_km
samples['log_distance'] = np.log10(samples['distance_km'] + 1)

print(f"distance range: {samples['distance_km'].min():.1f}–{samples['distance_km'].max():.1f} km")
print(f"mean distance: {samples['distance_km'].mean():.1f} ± {samples['distance_km'].std():.1f}")

print("\nstep 5: data balance check")
ratio = len(samples) / len(sources_filt)
print(f"sample:source ratio = {ratio:.2f}:1")

if ratio < 0.1:
    print("very few samples per source")
elif ratio > 10:
    print("very few sources per sample")
else:
    print("✓ ratio looks ok")

print("\n" + "=" * 70)
print("running analysis")
print("=" * 70)

results = []

for metric in metrics:
    print("\n" + "=" * 70)
    print(f"metric: {metric}")
    print("=" * 70)

    data = samples[['distance_km', 'log_distance', metric]].dropna()
    if len(data) < 50:
        print(f"skipping, only {len(data)} samples")
        continue

    X = data[['log_distance']].values
    y = data[metric].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nlinear regression")
    lin = LinearRegression().fit(X_tr, y_tr)
    y_lin = lin.predict(X_te)

    r2_lin = r2_score(y_te, y_lin)
    slope = lin.coef_[0]

    print(f"r2 = {r2_lin:.4f}")
    print(f"slope = {slope:.6f}")
    print("increases with distance" if slope > 0 else "decreases with distance")

    print("\ndecision tree")
    tree = DecisionTreeRegressor(max_depth=2, random_state=42).fit(X_tr, y_tr)
    y_tree = tree.predict(X_te)

    r2_tree = r2_score(y_te, y_tree)
    rmse = np.sqrt(mean_squared_error(y_te, y_tree))

    tp_log = tree.tree_.threshold[0] if tree.tree_.threshold[0] > 0 else tree.tree_.threshold[1]
    tp_km = (10 ** tp_log) - 1

    print(f"r2 = {r2_tree:.4f}")
    print(f"rmse = {rmse:.4f}")
    print(f"tipping point ≈ {tp_km:.2f} km")

    print("\nrandom forest check")
    forest = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
    forest.fit(X_tr, y_tr)
    y_rf = forest.predict(X_te)

    r2_rf = r2_score(y_te, y_rf)
    print(f"r2 = {r2_rf:.4f}")

    delta = r2_tree - r2_lin
    print(f"delta r2 = {delta:.3f}")

    results.append({
        'metric': metric,
        'n_samples': len(data),
        'r2_linear': r2_lin,
        'r2_tree': r2_tree,
        'r2_forest': r2_rf,
        'tipping_point_km': tp_km,
        'slope': slope,
        'delta': delta
    })

    data['zone'] = data['distance_km'].apply(lambda x: 'impact' if x < tp_km else 'recovery')
    print("\nzone stats")
    print(data.groupby('zone')[metric].agg(['mean', 'std', 'count']))

print("\n" + "=" * 70)
print("making plots")
print("=" * 70)

n = len(results)
fig, axes = plt.subplots(n, 3, figsize=(20, 5 * n))
if n == 1:
    axes = axes.reshape(1, -1)
for i, res in enumerate(results):
    metric = res['metric']
    data = samples[['distance_km', 'log_distance', metric]].dropna()

    X = data[['log_distance']].values
    y = data[metric].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    lin = LinearRegression().fit(X_tr, y_tr)
    tree = DecisionTreeRegressor(max_depth=2, random_state=42).fit(X_tr, y_tr)
    forest = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X_tr, y_tr)
    X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    ax1 = axes[i, 0]
    ax1.scatter(X_te, y_te, alpha=0.5, s=30, label='test data')
    ax1.plot(X_plot, lin.predict(X_plot), color='royalblue', linewidth=2, 
             label=f'linear fit (r2={res["r2_linear"]:.3f})')
    ax1.set_xlabel('log distance (log10 km + 1)', fontsize=11)
    ax1.set_ylabel(metric.replace('_', ' '), fontsize=11)
    ax1.set_title(f"traditional: {metric}\nslope: {res['slope']:.6f}", 
                  fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = axes[i, 1]
    ax2.scatter(X_te, y_te, alpha=0.5, s=30, label='test data')
    ax2.plot(X_plot, tree.predict(X_plot), color='forestgreen', linewidth=2, 
             label=f'tree pred (r2={res["r2_tree"]:.3f})')
    
    tp_log = np.log10(res['tipping_point_km'] + 1)
    ax2.axvline(tp_log, color='crimson', linestyle='--', linewidth=2, 
                label=f'threshold: {res["tipping_point_km"]:.1f} km')
    
    ax2.set_xlabel('log distance (log10 km + 1)', fontsize=11)
    ax2.set_ylabel(metric.replace('_', ' '), fontsize=11)
    ax2.set_title(f"ml detection: {metric}\ndelta r2: +{res['delta']:.3f}", 
                  fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax3 = axes[i, 2]
    ax3.scatter(X_te, y_te, alpha=0.4, s=30, color='gray', label='test data')
    ax3.plot(X_plot, forest.predict(X_plot), color='darkorange', linewidth=2.5, 
             label=f'forest pred (r2={res["r2_forest"]:.3f})')
    
    ax3.set_xlabel('log distance (log10 km + 1)', fontsize=11)
    ax3.set_ylabel(metric.replace('_', ' '), fontsize=11)
    ax3.set_title(f"Random Forest: {metric}\nMulti-variable Fit", 
                  fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('multi_metric_comparison.png', dpi=300, bbox_inches='tight')
print("saved multi_metric_comparison.png")

print("\nsaving outputs")
pd.DataFrame(results).to_csv('analysis_summary_all_metrics.csv', index=False)

out_cols = ['sample_id', 'latitude', 'longitude', 'distance_km', 'log_distance'] + metrics
samples[out_cols].to_csv('processed_data_multi_metrics.csv', index=False)

with open('detailed_results_multi_metrics.txt', 'w') as f:
    for r in results:
        f.write(str(r) + "\n")

print("analysis done")

