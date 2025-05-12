import json
from datetime import date, datetime

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
import pandas as pd
import numpy as np
import json
import re

from fullcam_client import FullCAMClient


def convert_for_json(obj):
    """Convert non-serializable objects to serializable ones"""
    if isinstance(obj, (pd.Timestamp | datetime | date)):
        return obj.isoformat()
    elif isinstance(obj, (Point)):
        return str(obj)  # Convert geometry to WKT string representation
    elif isinstance(obj, (np.integer)):
        return int(obj)
    elif isinstance(obj, (np.floating)):
        return float(obj)
    elif isinstance(obj, (np.ndarray)):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    elif isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(item) for item in obj]
    return obj

def create_geojson(df, output_filename='cea_points.geojson'):
    """
    Convert the results DataFrame to a GeoJSON file using GeoPandas.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing center_x and center_y columns to be converted to points
    output_filename : str
        Filename for the output GeoJSON file
    
    Returns:
    --------
    geopandas.GeoDataFrame
        The GeoDataFrame that was written to the GeoJSON file
    """
    # Filter out rows with null coordinates
    valid_df = df.dropna(subset=['center_x', 'center_y'])
    
    # Create Point geometries from longitude (center_x) and latitude (center_y)
    # Note: GeoJSON expects coordinates in [longitude, latitude] order
    geometries = [Point(x, y) for x, y in zip(valid_df['center_x'], valid_df['center_y'])]
    
    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(
        valid_df, 
        geometry=geometries,
        crs="EPSG:4326"  # WGS84 coordinate system
    )
    
    # Write to GeoJSON
    gdf.to_file(output_filename, driver='GeoJSON')
    print(f"GeoJSON file created: {output_filename}")
    
    return gdf

# Function to parse string representations of arrays into actual lists
def parse_array_string(array_str):
    try:
        # Use json.loads to parse the string as a JSON array
        return json.loads(array_str)
    except:
        # If json.loads fails, try using regex to extract values
        pattern = r'\[(.*?)\]'
        match = re.search(pattern, array_str)
        if match:
            values_str = match.group(1)
            values = [float(val.strip()) for val in values_str.split(',')]
            return values
        return []

def calculate_model_points(output_stats):

    df = pd.read_csv(output_stats)
    # Process each row
    results = []
    for _, row in df.iterrows():
        # Parse the string arrays into actual arrays
        center_x = parse_array_string(row['center_x'])
        center_y = parse_array_string(row['center_y'])
        values = parse_array_string(row['values'])
        
        # Get the mean value for this CEA
        mean_value = row['mean']
        
        # Filter for values greater than the mean
        filtered_data = [(val, i) for i, val in enumerate(values) if val >= mean_value]
        
        if filtered_data:
            # Find the value closest to the mean
            closest_value, closest_idx = min(filtered_data, key=lambda x: abs(x[0] - mean_value))
            
            # Get the corresponding center_x and center_y
            result = {
                'CEA': row['CEA'],
                'mean': mean_value,
                'closest_value': closest_value,
                'difference': abs(closest_value - mean_value),
                'center_x': center_x[closest_idx],
                'center_y': center_y[closest_idx]
            }
        else:
            # If no values are greater than the mean
            result = {
                'CEA': row['CEA'],
                'mean': mean_value,
                'closest_value': None,
                'difference': None,
                'center_x': None,
                'center_y': None
            }
        
        results.append(result)

    # Create a DataFrame with the results
    results_df = pd.DataFrame(results)

    # Display the results
    print(results_df[['CEA', 'mean', 'closest_value', 'difference', 'center_x', 'center_y']])

    # Save the results to a new CSV file if needed
    results_df.to_csv('cea_analysis_results.csv', index=False)

    return results_df


if __name__ == "__main__":
    # Initialize the FullCAM client
    client = FullCAMClient(version="2020")
    template = "ERF\\Environmental Plantings Method.plo"

    #sim_path = "C:\\Development\\MullionGroup\\Wollemi-Demo\\CEA-Stratification\\Manual Stratification to reduce CVs"
    #sim_path = "C:\\Development\\MullionGroup\\Wollemi-Demo\\CEA-Stratification\\Auto Stratification"
    #property_file = "BelokaCEAsMerged"
    #layer_property = "layer"

    #sim_path = "C:\\Development\\MullionGroup\\Wollemi-Demo\\CEA-Stratification\\Kalimna"
    #property_file = "Kalimna_ceas_area_remain_base"

    #sim_path = "C:\\Development\\MullionGroup\\Wollemi-Demo\\CEA-Stratification\\Kalimna_worst"
    #property_file = "Kalimna_ceas_area_remain_worst"

    #sim_path = "C:\\Development\\MullionGroup\\Wollemi-Demo\\CEA-Stratification\\Beloka_base"
    #property_file = "Beloka_ceas_area_remain_base"

    sim_path = "C:\\Development\\MullionGroup\\Wollemi-Demo\\CEA-Stratification\\Beloka_worst"
    property_file = "Beloka_ceas_area_remain_worst"


    layer_property = "CEA"
    model_points_method = "from_stats"  # or centroid or representative_point

    gdf = gpd.read_file(f"{sim_path}\\{property_file}.geojson")


    model_points_filename=f"{sim_path}\\{property_file}_representative_points.geojson"
    if model_points_method == "from_stats":
        model_points_df = calculate_model_points(f"{sim_path}\\output_stats.csv")
        model_points_gdf = create_geojson(model_points_df, output_filename=model_points_filename)
        gdf = pd.merge(gdf, model_points_df, on=layer_property, how='left')
    elif model_points_method == "centroid":
        model_points_gdf = gdf.geometry.centroid
        model_points_gdf.to_file(model_points_filename, driver="GeoJSON")
    elif model_points_method == "representative_point":
        model_points_gdf = gdf.geometry.representative_point()
        model_points_gdf.to_file(model_points_filename, driver="GeoJSON")

    projected_gdf = gdf.to_crs(epsg=3577)  # Australian Albers Equal Area
    projected_gdf["centroid"] = projected_gdf.geometry.centroid
    projected_gdf["area_ha_albers"] = projected_gdf.geometry.area / 10000
    
    all_results = []
    for i in range(len(gdf)):
        if model_points_method == "centroid":
            model_point = gdf.at[i, "geometry"].centroid
            center_x = model_point.x
            center_y = model_point.y
        elif model_points_method == "from_stats":
            center_x = gdf.at[i, "center_x"]
            center_y = gdf.at[i, "center_y"]
        elif model_points_method == "representative_point":
            model_point = gdf.at[i, "geometry"].representative_point()
            center_x = model_point.x
            center_y = model_point.y
        else:
            raise ValueError("Invalid model_points_method. Choose 'centroid', 'from_stats', or 'representative_point'.")

        area = projected_gdf.at[i, "area_ha_albers"]

        if "Plant Date" in gdf.columns:
            plant_date = gdf.at[i, "Plant Date"]
        else:
            plant_date = pd.Timestamp("2025-09-01")
            

        if "Configuration" in gdf.columns:
            config = gdf.at[i, "Configuration"]
        else:
            config = "EP"

        layer = gdf.at[i, layer_property]

        simulation = client.create_simulation_from_template(template, layer)

        notes = {
            "layer": layer,
            "area": area,
            "plant_date": plant_date.strftime("%Y-%m-%d"),
            "model_point": {"latitude": center_y, "longitude": center_x},
            "model_points_method": model_points_method,
            "configuration": config,
            "properties": convert_for_json(gdf.iloc[i].drop("geometry").to_dict()),
        }
        simulation.about.name = layer
        simulation.about.notes = json.dumps(notes)
        simulation.timing.use_daily_timing = False
        simulation.timing.start_date = plant_date
        simulation.timing.end_date = plant_date + pd.DateOffset(years=25)
        simulation.build.latitude = center_y
        simulation.build.longitude = center_x
        simulation.build.forest_category = "ERF"

        xml = client.get_location_xml(
            simulation.build.latitude,
            simulation.build.longitude,
            forest_category=simulation.build.forest_category,
        )

        species_list = simulation.apply_location_xml(xml)

        #find "Mixed species environmental planting" in the species list
        species = [
            species for species in species_list if "Mixed species environmental planting" in species["name"]
        ]
        # Get the first species in the list
        if species:
            env_planting = species[0]
        else:
            print(f"No species found for {layer}") 

        spec_xml = client.get_species_xml(
            simulation.build.latitude,
            simulation.build.longitude,            
            forest_category=simulation.build.forest_category,
            species_id=env_planting["id"])

        simulation.apply_species_xml(spec_xml, env_planting["id"], "Plant trees: Mixed species environmental planting on land managed for environmental services", plant_date)

        simulation.save_to_plo(f"{sim_path}/{layer}.plo")
        
        # Run the simulation
        results = simulation.run()
        simulation.save_csv(f"{sim_path}/{layer}.csv")
        df = simulation.to_dataframe()
        df["layer"] = layer
        df["area_ha"] = area
        all_results.append(df)

        
    # After the loop completes, concatenate all results
    combined_df = pd.concat(all_results, ignore_index=True)

    # Now you can work with the combined DataFrame
    print(f"Combined results: {len(combined_df)} rows")
    combined_df.to_csv(f"{sim_path}\\all_results.csv", index=False)
    #accus=('C mass of trees  (tC/ha)'+'C mass of forest debris  (tC/ha)')*'area_ha'*44/12*0.95
    combined_df['accus'] = (combined_df['C mass of trees  (tC/ha)'] + combined_df['C mass of forest debris  (tC/ha)']) * combined_df['area_ha'] * 44 / 12 * 0.95
    combined_df.to_csv(f"{sim_path}\\all_results_accus.csv", index=False)
    combined_df.to_parquet(f"{sim_path}/all_results.parquet", index=False)