
import pandas as pd
import numpy as np
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("IMPROVED µSudAqua DATA EXTRACTION")
print("Processing all 4 files for maximum coverage")
print("="*70)

print("\nSTEP 1: Loading metadata file...")

# The metadata file has coordinates and basic diversity info
metadata = pd.read_csv(r'C:\Users\letch\OneDrive\Documents\research\microsudaqua_metadata_V1.0_July2022.tsv', sep='\t')
print(f"✓ Loaded metadata: {len(metadata)} samples")
print(f"   Available columns: {list(metadata.columns)}")

# Extract core columns
required_cols = ['SampleID', 'Lat', 'Long']
if not all(col in metadata.columns for col in required_cols):
    print(f"ERROR: Missing required columns")
    print(f"   Expected: {required_cols}")
    print(f"   Found: {list(metadata.columns)}")
    raise ValueError("Metadata file structure unexpected")

microbiome = metadata[required_cols].copy()
microbiome.columns = ['sample_id', 'latitude', 'longitude']

# Remove samples without coordinates
before = len(microbiome)
microbiome = microbiome.dropna(subset=['latitude', 'longitude'])
print(f"✓ Kept {len(microbiome)} samples with valid coordinates (removed {before - len(microbiome)})")

print("\nSTEP 2: Loading RAW ASV abundance table...")

asv_file_path = r'C:\Users\letch\OneDrive\Documents\research\microsudaqua_rawtable_V1.0_July2022.tsv'

try:
    # Load and transpose immediately
    full_asv_df = pd.read_csv(asv_file_path, sep='\t', index_col=0)
    asv_table = full_asv_df.T 
    
    # This ensures Sample IDs are strings for the padding logic in Step 3
    asv_table.index = asv_table.index.astype(str)
    
    use_precalculated = False
    print(f"✓ Loaded Raw Table: {asv_table.shape[0]} samples found.")

except FileNotFoundError:
    print(f"CRITICAL ERROR: Could not find rawtable at {asv_file_path}")
    raise

print("\nSTEP 3: Calculating diversity metrics...")

if not use_precalculated and asv_table is not None:
    print("   Calculating Alpha Diversity (Shannon, Richness)...")
    
    diversity_results = []
    
    for raw_id in asv_table.index:
        # --- NEW: Standardize the ID before processing ---
        s = str(raw_id)
        if 'Micro_P' in s:
            parts = s.split('_')
            clean_id = f"Micro_P_{parts[-1].zfill(7)}"
        else:
            clean_id = f"Micro_P_{s.zfill(7)}"

        # Get counts for this sample and ensure they are numbers
        row_counts = pd.to_numeric(asv_table.loc[raw_id], errors='coerce').fillna(0)
        abundances = row_counts[row_counts > 0]
        
        if len(abundances) == 0:
            continue
        
        # 1. Richness
        richness = len(abundances)
        # 2. Shannon
        props = abundances / abundances.sum()
        shannon = -np.sum(props * np.log(props + 1e-9))
        # 3. Pielou's Evenness
        pielou = shannon / np.log(richness) if richness > 1 else 0
        
        diversity_results.append({
            'sample_id': clean_id,
            'species_richness': richness,
            'shannon_diversity': shannon,
            'pielou_evenness': pielou
        })
    
    diversity_df = pd.DataFrame(diversity_results)
    
    # DEBUG before merge
    print(f"   DEBUG - Diversity IDs (first 3): {diversity_df['sample_id'].head(3).tolist()}")
    print(f"   DEBUG - Metadata IDs (first 3): {microbiome['sample_id'].head(3).tolist()}")
    
    # MERGE
    microbiome = microbiome.merge(diversity_df, on='sample_id', how='inner')
    print(f"✓ Calculated and merged diversity for {len(microbiome)} samples")

else:
    print("   Skipping calculation; using pre-calculated metadata or file not found.")

print("\nSTEP 4: Adding environmental variables...")

env_vars = ['Temperature', 'pH', 'Conductivity', 'Oxygen', 'Nitrate', 
            'Phosphate', 'Salinity', 'Chlorophyll', 'Turbidity']

env_found = []
for var in env_vars:
    if var in metadata.columns:
        microbiome[var.lower()] = metadata.set_index('SampleID').loc[microbiome['sample_id'], var].values
        env_found.append(var)

if env_found:
    print(f"✓ Added environmental variables: {', '.join(env_found)}")
else:
    print("No environmental variables found in metadata")

print("\nSTEP 5: Processing taxonomy data...")
taxonomy_included = False 
filtered_data = None

taxonomy = None

# If we used the filtered file and it has taxonomy, use that
if taxonomy_included and filtered_data is not None:
    print("   Using taxonomy from filtered bacterial file...")
    try:
        tax_cols = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
        available_tax_cols = [col for col in tax_cols if col in filtered_data.columns]
        
        if available_tax_cols:
            # Create taxonomy dataframe with ASV IDs as index
            if 'ASV_ID' in filtered_data.columns:
                taxonomy = filtered_data[['ASV_ID'] + available_tax_cols].set_index('ASV_ID')
            elif filtered_data.columns[0] not in tax_cols:
                taxonomy = filtered_data[[filtered_data.columns[0]] + available_tax_cols].set_index(filtered_data.columns[0])
            
            if taxonomy is not None:
                print(f"   ✓ Taxonomy levels: {', '.join(available_tax_cols)}")
    except Exception as e:
        print(f"  Could not extract taxonomy from filtered file: {e}")
        taxonomy = None

# Otherwise try the separate taxonomy file
if taxonomy is None:
    print("   Trying separate taxonomy file...")
    taxonomy_files = [
        r'C:\Users\letch\OneDrive\Documents\research\microsudaqua_rawtaxonomy_blast_silva132_nr99_V1.0_July2022.tsv',
        'microsudaqua_rawtaxonomy_blast_silva132_nr99_V1.0_July2022.tsv',
        'rawtaxonomy_blast_silva132_nr99.tsv',
        'RawTaxonomy.tsv'
    ]
    
    for taxfile in taxonomy_files:
        try:
            taxonomy = pd.read_csv(taxfile, sep='\t', index_col=0)
            print(f"✓ Loaded taxonomy from: {taxfile}")
            print(f"   {len(taxonomy)} ASVs with taxonomy")
            break
        except FileNotFoundError:
            continue

# Extract dominant phyla if we have both taxonomy and ASV data
if taxonomy is not None and not use_precalculated and asv_table is not None:
    print("   Calculating dominant phyla per sample...")
    
    # Initialize the column first
    microbiome['dominant_phylum'] = None
    
    phylum_counts = []
    for sample_id in microbiome['sample_id']:
        if sample_id not in asv_table.index:
            continue
        
        sample_abundances = asv_table.loc[sample_id]
        top_asvs = sample_abundances.nlargest(10).index
        
        # Get phyla of top ASVs
        sample_phyla = {}
        for asv in top_asvs:
            if asv in taxonomy.index:
                phylum = str(taxonomy.loc[asv, 'Phylum']) if 'Phylum' in taxonomy.columns else 'Unknown'
                # Clean phylum name
                phylum = phylum.split(';')[-1].strip() if ';' in phylum else phylum.strip()
                sample_phyla[phylum] = sample_phyla.get(phylum, 0) + 1
        
        # Store dominant phylum
        if sample_phyla:
            dominant = max(sample_phyla, key=sample_phyla.get)
            microbiome.loc[microbiome['sample_id'] == sample_id, 'dominant_phylum'] = dominant
    
    # Check how many were added
    if 'dominant_phylum' in microbiome.columns:
        non_null = microbiome['dominant_phylum'].notna().sum()
        print(f"✓ Added dominant phylum for {non_null} samples")
    else:
        print("Could not add dominant phylum column")
else:
    if taxonomy is None:
        print(" No taxonomy file found - skipping phylum analysis")
    elif use_precalculated:
        print("   Skipping phylum calculation (using pre-calculated diversity)")
    elif asv_table is None:
        print("   Skipping phylum calculation (no ASV table available)")

print("\nSTEP 6: Quality filtering...")

before_qc = len(microbiome)

# DEBUG: Check what columns we actually have
print(f"\n   DEBUG: Current columns in microbiome: {list(microbiome.columns)}")
print(f"   DEBUG: Sample count before filtering: {len(microbiome)}")

# 1. Remove samples with missing diversity
if 'shannon_diversity' in microbiome.columns:
    before_diversity = len(microbiome)
    microbiome = microbiome.dropna(subset=['shannon_diversity'])
    print(f"   Removed {before_diversity - len(microbiome)} samples without shannon_diversity")

# 2. Extreme value filter
if 'shannon_diversity' in microbiome.columns:
    before_extreme = len(microbiome)
    microbiome = microbiome[(microbiome['shannon_diversity'] >= 0) & 
                            (microbiome['shannon_diversity'] <= 10)]
    print(f"   Removed {before_extreme - len(microbiome)} samples with extreme shannon values")
    
# 3. Low richness filter
if 'species_richness' in microbiome.columns:
    before_richness = len(microbiome)
    microbiome = microbiome[microbiome['species_richness'] >= 10]
    print(f"   Removed {before_richness - len(microbiome)} samples with richness < 10 ASVs")

# 4. FINAL DEDUPLICATION (ID ONLY, NOT COORDINATES)
# We keep multiple samples at the same lat/long, but remove if the exact SAME SampleID is there twice
before_dup = len(microbiome)
microbiome = microbiome.drop_duplicates(subset=['sample_id'], keep='first')
print(f"   Removed {before_dup - len(microbiome)} redundant SampleID entries")

print(f"✓ Final dataset: {len(microbiome)} high-quality samples")
print(f"   Quality control removed {before_qc - len(microbiome)} samples total")

print("\nSTEP 7: Analyzing spatial distribution...")

# Safety check - make sure we have valid coordinates
if microbiome['latitude'].isna().all() or microbiome['longitude'].isna().all():
    print("  ERROR: All latitude/longitude values are NaN!")
    print("   This means coordinates were lost during processing.")
    print(f"   Current columns: {list(microbiome.columns)}")
    print(f"   Sample count: {len(microbiome)}")
    raise ValueError("No valid coordinates remaining")

# Remove any remaining NaN coordinates
before_coord_filter = len(microbiome)
microbiome = microbiome.dropna(subset=['latitude', 'longitude'])
if before_coord_filter != len(microbiome):
    print(f"   Removed {before_coord_filter - len(microbiome)} samples with NaN coordinates")

if len(microbiome) == 0:
    raise ValueError("No samples with valid coordinates remaining")

lat_range = microbiome['latitude'].max() - microbiome['latitude'].min()
lon_range = microbiome['longitude'].max() - microbiome['longitude'].min()

print(f"   Latitude range: {microbiome['latitude'].min():.2f}° to {microbiome['latitude'].max():.2f}° (Δ={lat_range:.2f}°)")
print(f"   Longitude range: {microbiome['longitude'].min():.2f}° to {microbiome['longitude'].max():.2f}° (Δ={lon_range:.2f}°)")

# Calculate approximate spatial extent
from geopy.distance import geodesic
try:
    north_south_km = geodesic(
        (microbiome['latitude'].min(), microbiome['longitude'].mean()),
        (microbiome['latitude'].max(), microbiome['longitude'].mean())
    ).km
    east_west_km = geodesic(
        (microbiome['latitude'].mean(), microbiome['longitude'].min()),
        (microbiome['latitude'].mean(), microbiome['longitude'].max())
    ).km
    
    print(f"   Spatial extent: ~{north_south_km:.0f} km (N-S) × ~{east_west_km:.0f} km (E-W)")
except Exception as e:
    print(f" Could not calculate spatial extent: {e}")

print("\nSTEP 8: Exporting processed data...")

# Select final columns
export_cols = ['sample_id', 'latitude', 'longitude']

# Add diversity metrics
diversity_metrics = ['shannon_diversity', 'pielou_evenness', 'species_richness']
export_cols.extend([col for col in diversity_metrics if col in microbiome.columns])

# Add environmental variables
export_cols.extend([col for col in microbiome.columns if col not in export_cols 
                   and col not in ['dominant_phylum']])

# Add taxonomy if available
if 'dominant_phylum' in microbiome.columns:
    export_cols.append('dominant_phylum')

final_data = microbiome[export_cols].copy()
final_data.to_csv('microbiome_data.csv', index=False)

print(f"✓ Saved: 'microbiome_data.csv'")
print(f"   {len(final_data)} samples × {len(export_cols)} variables")
print(f"\n   Variables included:")
for col in export_cols:
    non_null = final_data[col].notna().sum()
    print(f"     • {col}: {non_null}/{len(final_data)} samples")

print("\n" + "="*70)
print("DATASET SUMMARY")
print("="*70)

print(f"\nSample Size: {len(final_data)} georeferenced microbiome samples")

if 'shannon_diversity' in final_data.columns:
    print(f"\nShannon Diversity:")
    print(f"   Range: {final_data['shannon_diversity'].min():.2f} - {final_data['shannon_diversity'].max():.2f}")
    print(f"   Mean ± SD: {final_data['shannon_diversity'].mean():.2f} ± {final_data['shannon_diversity'].std():.2f}")

if 'pielou_evenness' in final_data.columns:
    print(f"\nPielou's Evenness:")
    print(f"   Range: {final_data['pielou_evenness'].min():.2f} - {final_data['pielou_evenness'].max():.2f}")
    print(f"   Mean ± SD: {final_data['pielou_evenness'].mean():.2f} ± {final_data['pielou_evenness'].std():.2f}")

if 'species_richness' in final_data.columns:
    print(f"\nSpecies Richness:")
    print(f"   Range: {int(final_data['species_richness'].min())} - {int(final_data['species_richness'].max())} ASVs")
    print(f"   Mean ± SD: {final_data['species_richness'].mean():.0f} ± {final_data['species_richness'].std():.0f}")
print("DATA PREPARATION COMPLETE")
