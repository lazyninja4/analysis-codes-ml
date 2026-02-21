import pandas as pd
import numpy as np
import requests
import time
from geopy.distance import geodesic

print("rigorous filter-based pollution source extraction")

print("STEP 1: Querying OpenStreetMap (with filters)...")

def query_osm_strict(region_name, bbox):
    """
    query ONLY wastewater treatment plants (not pumping stations)
    bbox: [south, west, north, east]
    """
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    query = f"""
    [out:json][timeout:180];
    (
      node["man_made"="wastewater_plant"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
      way["man_made"="wastewater_plant"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
      relation["man_made"="wastewater_plant"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
    );
    out center;
    """
    
    print(f"  Querying {region_name}...", end=" ")
    try:
        response = requests.post(overpass_url, data={'data': query}, timeout=300)
        response.raise_for_status()
        data = response.json()
        
        plants = []
        for element in data['elements']:
            tags = element.get('tags', {})
            
            if any(x in str(tags).lower() for x in ['pump', 'storm', 'drainage', 'septic']):
                continue
            
            if element['type'] == 'node':
                plants.append({
                    'source_name': tags.get('name', f"WWTP_{element['id']}"),
                    'latitude': element['lat'],
                    'longitude': element['lon'],
                    'osm_id': element['id'],
                    'region': region_name
                })
            elif element['type'] in ['way', 'relation'] and 'center' in element:
                plants.append({
                    'source_name': tags.get('name', f"WWTP_{element['id']}"),
                    'latitude': element['center']['lat'],
                    'longitude': element['center']['lon'],
                    'osm_id': element['id'],
                    'region': region_name
                })
        
        print(f"✓ {len(plants)} plants")
        return pd.DataFrame(plants)
        
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()

priority_regions = {
    'Brazil-Southeast': [-24, -50, -19, -39],  # São Paulo, Rio area
    'Brazil-South': [-34, -54, -24, -48],      # Porto Alegre, Curitiba
    'Brazil-Central': [-17, -50, -12, -42],    # Brasília region
    'Argentina-Central': [-38, -65, -30, -56], # Buenos Aires, Córdoba
    'Colombia-Central': [1, -77, 8, -71],      # Bogotá, Medellín
    'Peru-Coast': [-18, -79, -10, -75],        # Lima, Arequipa
    'Chile-Central': [-40, -74, -30, -69],     # Santiago, Valparaíso
}

print("\nDownloading from OpenStreetMap (7 priority regions)...")
print("This may take 3-5 minutes total...")

all_plants = []
for i, (region_name, bbox) in enumerate(priority_regions.items(), 1):
    print(f"[{i}/7] ", end="")
    df = query_osm_strict(region_name, bbox)
    if not df.empty:
        all_plants.append(df)
    time.sleep(2) 

if not all_plants:
    print("\n=No OSM data downloaded. Using city proxy instead.")
    osm_plants = pd.DataFrame()
else:
    osm_plants = pd.concat(all_plants, ignore_index=True)
    print(f"\n✓ Downloaded {len(osm_plants)} plants from OSM")

print("\nSTEP 2: Deduplication and quality filtering...")

if not osm_plants.empty:
    before = len(osm_plants)
    
    osm_plants = osm_plants.drop_duplicates(subset=['latitude', 'longitude'])
    print(f"Removed {before - len(osm_plants)} exact duplicates")
    
    def remove_nearby_duplicates(df, min_distance_km=5):
        """Keep only one plant per min_distance_km radius"""
        coords = df[['latitude', 'longitude']].values
        kept_indices = [0] 
        
        for i in range(1, len(coords)):
            is_unique = True
            for j in kept_indices:
                dist = geodesic(coords[i], coords[j]).km
                if dist < min_distance_km:
                    is_unique = False
                    break
            if is_unique:
                kept_indices.append(i)
        
        return df.iloc[kept_indices].reset_index(drop=True)
    
    before = len(osm_plants)
    osm_plants = remove_nearby_duplicates(osm_plants, min_distance_km=5)
    print(f"Removed {before - len(osm_plants)} near-duplicates (<5km apart)")
    
    print(f"\nOSM plants after filtering: {len(osm_plants)}")

print("\nSTEP 3: Adding major city pollution proxies...")

major_cities = [
    {'source_name': 'São Paulo Metro', 'latitude': -23.5505, 'longitude': -46.6333},
    {'source_name': 'Rio de Janeiro Metro', 'latitude': -22.9068, 'longitude': -43.1729},
    {'source_name': 'Buenos Aires Metro', 'latitude': -34.6037, 'longitude': -58.3816},
    {'source_name': 'Bogotá Metro', 'latitude': 4.7110, 'longitude': -74.0721},
    {'source_name': 'Lima Metro', 'latitude': -12.0464, 'longitude': -77.0428},
    {'source_name': 'Santiago Metro', 'latitude': -33.4489, 'longitude': -70.6693},
    {'source_name': 'Brasília', 'latitude': -15.7942, 'longitude': -47.8822},
    {'source_name': 'Belo Horizonte', 'latitude': -19.9167, 'longitude': -43.9345},
    {'source_name': 'Porto Alegre', 'latitude': -30.0346, 'longitude': -51.2177},
    {'source_name': 'Medellín', 'latitude': 6.2476, 'longitude': -75.5658},
    {'source_name': 'Córdoba AR', 'latitude': -31.4201, 'longitude': -64.1888},
    {'source_name': 'Cali', 'latitude': 3.4516, 'longitude': -76.5320},
    {'source_name': 'Montevideo', 'latitude': -34.9011, 'longitude': -56.1645},
    {'source_name': 'Guayaquil', 'latitude': -2.1894, 'longitude': -79.8890},
    {'source_name': 'Quito', 'latitude': -0.1807, 'longitude': -78.4678},
]

city_plants = pd.DataFrame(major_cities)
print(f"dded {len(city_plants)} major urban centers as pollution proxies")

print("\nSTEP 4: Combining pollution sources...")

if osm_plants.empty:
    print("  Using only city proxies (OSM download failed)")
    final_pollution = city_plants
else:
    combined = pd.concat([osm_plants[['source_name', 'latitude', 'longitude']], 
                         city_plants], ignore_index=True)
    
    print("Removing OSM plants near cities (already covered by city proxies)...")
    
    city_coords = city_plants[['latitude', 'longitude']].values
    osm_coords = osm_plants[['latitude', 'longitude']].values
    
    keep_osm = []
    for i, osm_coord in enumerate(osm_coords):
        is_far_from_cities = True
        for city_coord in city_coords:
            if geodesic(osm_coord, city_coord).km < 10:
                is_far_from_cities = False
                break
        if is_far_from_cities:
            keep_osm.append(i)
    
    filtered_osm = osm_plants.iloc[keep_osm][['source_name', 'latitude', 'longitude']]
    
    final_pollution = pd.concat([filtered_osm, city_plants], ignore_index=True)
    print(f" Kept {len(filtered_osm)} OSM plants (not near cities)")
    print(f"Kept {len(city_plants)} city proxies")

print(f"\nFINAL: {len(final_pollution)} pollution sources")

print("\nSTEP 5: Saving pollution sources...")

final_pollution.to_csv('pollution_sources.csv', index=False)
print("Saved: 'pollution_sources.csv'")

lat_range = final_pollution['latitude'].max() - final_pollution['latitude'].min()
lon_range = final_pollution['longitude'].max() - final_pollution['longitude'].min()

print(f"\nSpatial Coverage:")
print(f"   Latitude range: {final_pollution['latitude'].min():.2f}° to {final_pollution['latitude'].max():.2f}° (Δ={lat_range:.2f}°)")
print(f"   Longitude range: {final_pollution['longitude'].min():.2f}° to {final_pollution['longitude'].max():.2f}° (Δ={lon_range:.2f}°)")

from scipy.spatial import ConvexHull
coords = final_pollution[['longitude', 'latitude']].values
if len(coords) >= 4:
    hull = ConvexHull(coords)
    area_deg2 = hull.volume 
    area_km2 = area_deg2 * (111**2)
    density = len(final_pollution) / area_km2
    print(f"   Spatial density: {density:.4f} sources per 1000 km²")

print("\n" + "="*70)
print("POLLUTION SOURCE EXTRACTION COMPLETE!")
print("="*70)

print(f"\nCreated: pollution_sources.csv")
print(f"   {len(final_pollution)} pollution sources")
print(f"   Breakdown:")
if not osm_plants.empty:
    print(f"     • {len(filtered_osm)} from OpenStreetMap (filtered)")
print(f"     • {len(city_plants)} major urban centers")

print(f"\nTarget ratio check:")
print(f"   If you have ~700 microbiome samples:")
print(f"   Sample:Source ratio = {700/len(final_pollution):.2f}:1")
if 700/len(final_pollution) < 0.5:
    print(f" Ratio < 0.5:1 (too few samples per source)")
elif 700/len(final_pollution) > 10:
    print(f"  Ratio > 10:1 (too few pollution sources)")
else:
    print(f"Ratio is in optimal range (0.5:1 to 10:1)")
