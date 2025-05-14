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
from shapely.geometry import Point, LineString, box


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


def create_geojson(df, output_filename="cea_points.geojson"):
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
    valid_df = df.dropna(subset=["center_x", "center_y"])

    # Create Point geometries from longitude (center_x) and latitude (center_y)
    # Note: GeoJSON expects coordinates in [longitude, latitude] order
    geometries = [Point(x, y) for x, y in zip(valid_df["center_x"], valid_df["center_y"])]

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(
        valid_df,
        geometry=geometries,
        crs="EPSG:4326",  # WGS84 coordinate system
    )

    # Write to GeoJSON
    gdf.to_file(output_filename, driver="GeoJSON")
    print(f"GeoJSON file created: {output_filename}")

    return gdf


# Function to parse string representations of arrays into actual lists
def parse_array_string(array_str):
    try:
        # Use json.loads to parse the string as a JSON array
        return json.loads(array_str)
    except:
        # If json.loads fails, try using regex to extract values
        pattern = r"\[(.*?)\]"
        match = re.search(pattern, array_str)
        if match:
            values_str = match.group(1)
            values = [float(val.strip()) for val in values_str.split(",")]
            return values
        return []


def calculate_model_points(output_stats):
    df = pd.read_csv(output_stats)
    # Process each row
    results = []
    for _, row in df.iterrows():
        # Parse the string arrays into actual arrays
        center_x = parse_array_string(row["center_x"])
        center_y = parse_array_string(row["center_y"])
        values = parse_array_string(row["values"])
        mean_value = row["mean"]

        # Find the absolute differences between each value and the mean
        differences = np.abs(np.array(values) - mean_value)

        # Get the index of the minimum difference
        closest_idx = np.argmin(differences)

        closest_value = values[closest_idx]
        # Get the corresponding center_x and center_y
        result = {
            "CEA": row["CEA"],
            "mean": mean_value,
            "closest_value": closest_value,
            "difference": abs(closest_value - mean_value),
            "center_x": center_x[closest_idx],
            "center_y": center_y[closest_idx],
        }

        results.append(result)

    # Create a DataFrame with the results
    return pd.DataFrame(results)


def ensure_point_in_polygon(row, cell_size=0.025):
    """
    Check if a point is within its polygon. If not, but its cell intersects the polygon,
    modify the point to be within the polygon using representative_point().
    """
    point = row["point_geometry"]
    polygon = row["geometry"]  # The original polygon geometry from gdf

    # If the point is already in the polygon, return it as is
    if polygon.contains(point):
        return point

    # Create a box (cell) around the point
    x, y = point.x, point.y
    cell = box(x - cell_size / 2, y - cell_size / 2, x + cell_size / 2, y + cell_size / 2)

    # Check if the cell intersects with the polygon
    if cell.intersects(polygon):
        # Get the intersection between the cell and the polygon
        intersection = cell.intersection(polygon)

        # Use representative_point to get a point guaranteed to be within the intersection
        return intersection.representative_point()

    # If the cell doesn't intersect the polygon, return the original point
    return point


def find_visual_center(polygon):
    """Find a visual center avoiding holes"""

    # 1. Sample points along the exterior boundary
    num_samples = 50
    if polygon.geom_type == "MultiPolygon":
        # For MultiPolygons, get the largest polygon
        largest = max(polygon.geoms, key=lambda x: x.area)
        sample_poly = largest
        exterior_coords = [list(largest.exterior.coords)]
    else:
        sample_poly = polygon
        exterior_coords = list(polygon.exterior.coords)

    sample_step = max(1, len(exterior_coords) // num_samples)
    boundary_samples = exterior_coords[::sample_step]

    # 2. Calculate centroid of these sample points (not the polygon centroid)
    sample_points = np.array(boundary_samples)
    sample_centroid = Point(np.mean(sample_points[:, 0]), np.mean(sample_points[:, 1]))

    # 3. If the sample centroid is in the polygon and not in a hole, use it
    if sample_poly.contains(sample_centroid):
        return sample_centroid

    # 4. Otherwise, find the nearest point in the polygon to this centroid
    if not sample_poly.contains(sample_centroid):
        # Create a line from centroid to the nearest point on the boundary
        nearest_point = sample_poly.boundary.interpolate(
            sample_poly.boundary.project(sample_centroid)
        )
        line = LineString([sample_centroid, nearest_point])

        # Find a point slightly inside from the boundary
        if line.length > 0:
            inside_point = line.interpolate(line.length * 0.9)
            if sample_poly.contains(inside_point):
                return inside_point

    # Last resort - use representative point
    return sample_poly.representative_point()


if __name__ == "__main__":
    # Initialize the FullCAM client
    client = FullCAMClient(version="2020")
    template = "ERF\\Environmental Plantings Method.plo"

    # sim_path = "C:\\Development\\MullionGroup\\Wollemi-Demo\\CEA-Stratification\\Manual Stratification to reduce CVs"
    # sim_path = "C:\\Development\\MullionGroup\\Wollemi-Demo\\CEA-Stratification\\Auto Stratification"
    # property_file = "BelokaCEAsMerged"
    # layer_property = "layer"

    # sim_path = "C:\\Development\\MullionGroup\\Wollemi-Demo\\CEA-Stratification\\Kalimna"
    # property_file = "Kalimna_ceas_area_remain_base"

    # sim_path = "C:\\Development\\MullionGroup\\Wollemi-Demo\\CEA-Stratification\\Kalimna_worst"
    # property_file = "Kalimna_ceas_area_remain_worst"

    # sim_path = "C:\\Development\\MullionGroup\\Wollemi-Demo\\CEA-Stratification\\Beloka_base"
    # property_file = "Beloka_ceas_area_remain_base"
    # sim_path = "C:\\Development\\MullionGroup\\Wollemi-Demo\\CEA-Stratification\\Beloka_worst"
    # property_file = "Beloka_ceas_area_remain_worst"

    # Best Cases (Beloka and Kalimna): Plot location is the centroid
    # Base Case (Kalimna): Plot location as we did yesterday
    # Planting date is always 1 September and the respective year (2025 or 2026)

    # file_path = "C:\\Development\\MullionGroup\\Wollemi-Demo\\CEA-Stratification\\Kalimna_13May_2025\\Scenario_Base_2025_planting\\Shapefiles\\Kalimna_ceas_area_remain_base_thisyear.geojson"
    # file_path =  "C:\\Development\\MullionGroup\\Wollemi-Demo\\CEA-Stratification\\Kalimna_13May_2025\\Scenario_Base_2026_planting\\Shapefiles\\Kalimna_ceas_area_remain_base_nextyear.geojson"
    # file_path = "C:\\Development\\MullionGroup\\Wollemi-Demo\\CEA-Stratification\\Kalimna_13May_2025\\Scenario_Best_2025_planting\\Shapefiles\\Kalimna_ceas_area_remain_best_thisyear.geojson"
    # file_path = "C:\\Development\\MullionGroup\\Wollemi-Demo\\CEA-Stratification\\Kalimna_13May_2025\\Scenario_Best_2026_planting\\Shapefiles\\Kalimna_ceas_area_remain_best_nextyear.geojson"

    file_path = "C:\\Development\\MullionGroup\\Wollemi-Demo\\CEA-Stratification\\Beloka_13May_2025\\Scenario_Best\\Shapefiles\\Beloka_ceas_area_remain_best_2.geojson"

    sim_path = file_path.split("\\Shapefiles")[0]
    property_file = file_path.split("\\Shapefiles\\")[-1].split(".")[0]

    layer_property = "CEA"
    model_points_method = "from_stats"  # or centroid or visual_center

    if property_file.find("_best_") > 0:
        model_points_method = "centroid"
    elif property_file.find("_base_") > 0:
        model_points_method = "from_stats"

    gdf = gpd.read_file(f"{sim_path}\\Shapefiles\\{property_file}.geojson")

    model_points_filename = (
        f"{sim_path}\\FullCAM_ModelPoints\\{property_file}_representative_points.geojson"
    )

    if model_points_method == "from_stats":
        model_points_df = calculate_model_points(
            f"{sim_path}\\FullCAM_ModelPoints\\output_stats.csv"
        )
        model_points_gdf = pd.merge(gdf, model_points_df, on=layer_property, how="left")

        # First create points from center coordinates
        model_points_gdf["point_geometry"] = model_points_gdf.apply(
            lambda row: Point(row["center_x"], row["center_y"]), axis=1
        )

        # Apply the function to create adjusted points that are within their polygons
        model_points_gdf["geometry"] = model_points_gdf.apply(
            ensure_point_in_polygon, cell_size=0.025, axis=1
        )

        # Drop the temporary point geometry column
        model_points_gdf.drop(columns=["point_geometry"], inplace=True)

    elif model_points_method == "centroid":
        model_points_gdf = gdf.copy()
        model_points_gdf.geometry = gdf.geometry.centroid
    elif model_points_method == "visual_center":
        model_points_gdf = gdf.copy()
        model_points_gdf.geometry = gdf.geometry.apply(find_visual_center)
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
        elif model_points_method == "visual_center":
            model_point = find_visual_center(gdf.at[i, "geometry"])
            center_x = model_point.x
            center_y = model_point.y
        else:
            raise ValueError(
                "Invalid model_points_method. Choose 'centroid', 'from_stats', or 'representative_point'."
            )

        area = projected_gdf.at[i, "area_ha_albers"]

        if "Plant Date" in gdf.columns:
            plant_date = gdf.at[i, "Plant Date"]
        elif "Plant" in gdf.columns:
            year = int(gdf.at[i, "Plant"])
            plant_date = pd.Timestamp(f"{year}-09-01")
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

        # find "Mixed species environmental planting" in the species list
        species = [
            species
            for species in species_list
            if "Mixed species environmental planting" in species["name"]
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
            species_id=env_planting["id"],
        )

        simulation.apply_species_xml(
            spec_xml,
            env_planting["id"],
            "Plant trees: Mixed species environmental planting on land managed for environmental services",
            plant_date,
        )

        simulation.save_to_plo(f"{sim_path}/FullCAM_Plotfiles/{layer}.plo")

        # Run the simulation
        results = simulation.run()
        simulation.save_csv(f"{sim_path}/FullCAM_Ouput/{layer}.csv")
        df = simulation.to_dataframe()
        df["layer"] = layer
        df["area_ha"] = area
        all_results.append(df)

    # After the loop completes, concatenate all results
    combined_df = pd.concat(all_results, ignore_index=True)

    # Now you can work with the combined DataFrame
    print(f"Combined results: {len(combined_df)} rows")
    combined_df.to_csv(f"{sim_path}/FullCAM_Ouput/all_results.csv", index=False)
    combined_df.to_parquet(f"{sim_path}/FullCAM_Ouput/all_results.parquet", index=False)
    # accus=('C mass of trees  (tC/ha)'+'C mass of forest debris  (tC/ha)')*'area_ha'*44/12*0.95
    # combined_df['accus'] = (combined_df['C mass of trees  (tC/ha)'] + combined_df['C mass of forest debris  (tC/ha)']) * combined_df['area_ha'] * 44 / 12 * 0.95
    # combined_df.to_csv(f"{sim_path}\\all_results_accus.csv", index=False)
