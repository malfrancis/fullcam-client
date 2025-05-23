import json
import re
from datetime import date, datetime

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, box

from fullcam_client import FullCAMClient
import exactextract

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


def calculate_model_points(output_stats, method="less_than_mean"):
    df = pd.read_csv(output_stats)
    # Process each row
    results = []
    for _, row in df.iterrows():
        # Parse the string arrays into actual arrays
        center_x = parse_array_string(row["center_x"])
        center_y = parse_array_string(row["center_y"])
        values = parse_array_string(row["values"])

        if method == "less_than_mean":
            target_value = row["mean"]
        elif method == "median":
            target_value = row["median"]
        else:
            raise ValueError("Invalid method. Choose 'less_than_mean' or 'median'.")

        # Find the absolute differences between each value and the target value
        differences = np.abs(np.array(values) - target_value)

        # Filter values less than or equal to the target value
        valid_indices = [i for i, v in enumerate(values) if v <= target_value]

        if valid_indices:
            # Get the index of the closest valid value
            closest_idx = min(valid_indices, key=lambda i: differences[i])

            closest_value = values[closest_idx]
            # Get the corresponding center_x and center_y
            result = {
                "CEA": row["CEA"],
                "target_value": target_value,
                "closest_value": closest_value,
                "difference": abs(closest_value - target_value),
                "center_x": center_x[closest_idx],
                "center_y": center_y[closest_idx],
            }

            results.append(result)

    # Create a DataFrame with the results
    return pd.DataFrame(results)


def calculate_model_pixels(gdf, layer_property, m_layer):
    #df = pd.read_csv(output_stats)

    # Get area-weighted statistics with exact coverage
    df = exactextract.exact_extract(
        m_layer,
        gdf,
        ["cell_id", "mean", "sum", "count", "center_x", "center_y", "coverage", "values"],
        include_cols=[layer_property],
        output="pandas",
    )

    return df.explode(['cell_id','center_x', 'center_y', 'coverage', 'values'])


def ensure_point_in_polygon(row, cell_size=0.0025):
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
    pad = cell_size / 2
    # Create a box around the point
    cell = box(x - pad, y - pad, x + pad, y + pad)
    # print(f"Cell: {cell.wkt}")
    # Check if the cell intersects with the polygon
    if cell.intersects(polygon):
        # Get the intersection between the cell and the polygon
        intersection = cell.intersection(polygon)
        # print(f"Intersection: {intersection.wkt}")
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


# "C:\Development\MullionGroup\Wollemi-Demo\CEA-Stratification\_Scenarios_15May_2025_Talbot_KangarooC\KangarooCamp\Scenario_Base\Shapefiles\KangarooCamp_plantable_area_mid.geojson"
def simulate_cea(
    file_path,
    layer_property="CEA",
    model_points_method="from_stats",  # from_stats or centroid or visual_center
    method="median",
):
    client = FullCAMClient(version="2020")
    template = "ERF\\Environmental Plantings Method.plo"
    sim_path = file_path.split("\\Shapefiles")[0]
    property_file = file_path.split("\\Shapefiles\\")[-1].split(".")[0]

    gdf = gpd.read_file(file_path)

    model_points_filename = (
        f"{sim_path}\\FullCAM_ModelPoints\\{property_file}_representative_points.geojson"
    )

    if model_points_method == "from_stats":
        model_points_df = calculate_model_points(
            f"{sim_path}\\FullCAM_ModelPoints\\output_stats.csv", method=method
        )

        model_points_gdf = pd.merge(gdf, model_points_df, on=layer_property, how="left")

        # First create points from center coordinates
        model_points_gdf["point_geometry"] = model_points_gdf.apply(
            lambda row: Point(row["center_x"], row["center_y"]), axis=1
        )

        # Apply the function to create adjusted points that are within their polygons
        model_points_gdf["geometry"] = model_points_gdf.apply(
            ensure_point_in_polygon, cell_size=0.0025, axis=1
        )

        # Drop the temporary point geometry column
        model_points_gdf.drop(columns=["point_geometry"], inplace=True)

    elif model_points_method == "centroid":
        model_points_gdf = gdf.copy()
        model_points_gdf.geometry = gdf.geometry.centroid
    elif model_points_method == "visual_center":
        model_points_gdf = gdf.copy()
        model_points_gdf.geometry = gdf.geometry.apply(find_visual_center)

    model_points_gda94_gdf = model_points_gdf.to_crs(epsg=4283)  # GDA94
    model_points_gda94_gdf.to_file(model_points_filename, driver="GeoJSON")
    # model_points_gda2020_gdf = model_points_gdf.to_crs(epsg=7844) # GDA2020

    projected_gdf = gdf.to_crs(epsg=3577)  # Australian Albers Equal Area
    projected_gdf["centroid"] = projected_gdf.geometry.centroid
    projected_gdf["area_ha_albers"] = projected_gdf.geometry.area / 10000

    all_results = []

    for idx, row in model_points_gda94_gdf.iterrows():
        model_point = row["geometry"]
        center_x = model_point.x
        center_y = model_point.y

        area = projected_gdf.loc[
            projected_gdf[layer_property] == row[layer_property], "area_ha_albers"
        ].values[0]

        if "Plant Date" in gdf.columns:
            plant_date = row["Plant Date"]
        elif "Plant" in gdf.columns:
            year = int(row["Plant"])
            plant_date = pd.Timestamp(f"{year}-09-01")
        else:
            plant_date = pd.Timestamp("2025-09-01")

        if "Configuration" in gdf.columns:
            config = row["Configuration"]
        else:
            config = "EP"

        layer = row[layer_property]

        simulation = client.create_simulation_from_template(template, layer)
        simulation.about.name = layer
        simulation.timing.use_daily_timing = False
        simulation.timing.start_date = plant_date
        simulation.timing.end_date = plant_date + pd.DateOffset(years=25)
        simulation.build.latitude = center_y
        simulation.build.longitude = center_x
        simulation.build.forest_category = "ERF"
        simulation.download_location_info()
        long_term_average_FPI = simulation.location_info.long_term_average_FPI
        maximum_aboveground_biomass = simulation.location_info.maximum_aboveground_biomass
        m_value = row["closest_value"]
        if not np.isclose(m_value, maximum_aboveground_biomass):
            print(
                f"Warning: m_value {m_value} does not match maximum_aboveground_biomass {maximum_aboveground_biomass} for {layer}"
            )
        FPI = simulation.location_info.forest_productivity_index["raw_values"]
        # compute tha average FPI
        average_FPI = np.mean(FPI)

        notes = {
            "layer": layer,
            "area": area,
            "plant_date": plant_date.strftime("%Y-%m-%d"),
            "model_point": {"latitude": center_y, "longitude": center_x},
            "model_points_method": "model_pixels",
            "configuration": config,
            "long_term_average_FPI": long_term_average_FPI,
            "maximum_aboveground_biomass": maximum_aboveground_biomass,
            "average_FPI": average_FPI,
            "properties": convert_for_json(row.drop("geometry").to_dict()),
        }
        simulation.about.notes = json.dumps(notes)

        env_planting = next(
            (
                species
                for species in simulation.location_info.forest_species
                if "Mixed species environmental planting" in species.name
            ),
            None,  # Default value if not found
        )
        if env_planting is None:
            print(f"No species found for {layer}")

        else:
            spec_xml = client.get_species_xml(
                simulation.build.latitude,
                simulation.build.longitude,
                forest_category=simulation.build.forest_category,
                species_id=env_planting.id,
            )

            simulation.apply_species_xml(
                spec_xml,
                env_planting.id,
                "Plant trees: Mixed species environmental planting on land managed for environmental services",
                plant_date,
            )

        simulation.save_to_plo(f"{sim_path}/FullCAM_Plotfiles/{layer}.plo")

        # Run the simulation
        simulation.run()
        simulation.save_csv(f"{sim_path}/FullCAM_Ouput/{layer}.csv")
        df = simulation.to_dataframe()
        df["layer"] = layer
        df["area_ha"] = area
        all_results.append(df)

    # After the loop completes, concatenate all results
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(f"{sim_path}/FullCAM_Ouput/all_results.csv", index=False)
    combined_df.to_parquet(f"{sim_path}/FullCAM_Ouput/all_results.parquet", index=False)

    return combined_df


def simulate_cea_cells(file_path, layer_property="CEA", m_layer=None):
    client = FullCAMClient(version="2020")
    template = "ERF\\Environmental Plantings Method.plo"
    sim_path = file_path.split("\\Shapefiles")[0]
    property_file = file_path.split("\\Shapefiles\\")[-1].split(".")[0]

    gdf = gpd.read_file(file_path)

    model_points_filename = (
        f"{sim_path}\\FullCAM_ModelPoints\\{property_file}_representative_points.geojson"
    )

    pixels_df = calculate_model_pixels(gdf, layer_property=layer_property, m_layer=m_layer)

    model_points_gdf = pd.merge(gdf, pixels_df, on=layer_property, how="left")

    model_points_gdf["point_geometry"] = model_points_gdf.apply(
        lambda row: Point(row["center_x"], row["center_y"]), axis=1
    )

    # Apply the function to create adjusted points that are within their polygons
    model_points_gdf["geometry"] = model_points_gdf.apply(
        ensure_point_in_polygon, cell_size=0.0025, axis=1
    )

    # Drop the temporary point geometry column
    model_points_gdf.drop(columns=["point_geometry"], inplace=True)

    model_points_gda94_gdf = model_points_gdf.to_crs(epsg=4283)  # GDA94
    model_points_gda94_gdf.to_file(model_points_filename, driver="GeoJSON")
    
    # model_points_gda2020_gdf = model_points_gdf.to_crs(epsg=7844) # GDA2020
    # projected_gda2020_gdf = gdf.to_crs('EPSG:9473') # Australian Albers GDA2020

    projected_gdf = gdf.to_crs(epsg=3577)  # Australian Albers Equal Area
    projected_gdf["area_ha_albers"] = projected_gdf.geometry.area / 10000

    all_results = []
    for idx, row in model_points_gda94_gdf.iterrows():
        model_point = row["geometry"]
        center_x = model_point.x
        center_y = model_point.y
        plot_idx = row["cell_id"]
        coverage = row["coverage"]
        count = row["count"]
        area = projected_gdf.loc[
            projected_gdf[layer_property] == row[layer_property], "area_ha_albers"
        ].values[0]

        area *= (coverage / count)

        if "Plant Date" in gdf.columns:
            plant_date = row["Plant Date"]
        elif "Plant" in gdf.columns:
            year = int(row["Plant"])
            plant_date = pd.Timestamp(f"{year}-09-01")
        else:
            plant_date = pd.Timestamp("2025-09-01")

        if "Configuration" in gdf.columns:
            config = row["Configuration"]
        else:
            config = "EP"

        layer = row[layer_property]

        simulation = client.create_simulation_from_template(template, layer)

        simulation.about.name = layer
        simulation.timing.use_daily_timing = False
        simulation.timing.start_date = plant_date
        simulation.timing.end_date = plant_date + pd.DateOffset(years=25)
        simulation.build.latitude = center_y
        simulation.build.longitude = center_x
        simulation.build.forest_category = "ERF"
        simulation.download_location_info()
        long_term_average_FPI = simulation.location_info.long_term_average_FPI
        maximum_aboveground_biomass = simulation.location_info.maximum_aboveground_biomass
        m_value = row["values"]
        if not np.isclose(m_value, maximum_aboveground_biomass):
            print(
                f"Warning: m_value {m_value} does not match maximum_aboveground_biomass {maximum_aboveground_biomass} for {layer}"
            )
        FPI = simulation.location_info.forest_productivity_index["raw_values"]
        # compute tha average FPI
        average_FPI = np.mean(FPI)

        notes = {
            "layer": layer,
            "area": area,
            "plant_date": plant_date.strftime("%Y-%m-%d"),
            "model_point": {"latitude": center_y, "longitude": center_x},
            "model_points_method": "model_pixels",
            "configuration": config,
            "long_term_average_FPI": long_term_average_FPI,
            "maximum_aboveground_biomass": maximum_aboveground_biomass,
            "average_FPI": average_FPI,
            "properties": convert_for_json(row.drop("geometry").to_dict()),
        }
        simulation.about.notes = json.dumps(notes)

        env_planting = next(
            (
                species
                for species in simulation.location_info.forest_species
                if "Mixed species environmental planting" in species.name
            ),
            None,  # Default value if not found
        )
        if env_planting is None:
            print(f"No species found for {layer}")

        else:
            spec_xml = client.get_species_xml(
                simulation.build.latitude,
                simulation.build.longitude,
                forest_category=simulation.build.forest_category,
                species_id=env_planting.id,
            )

            simulation.apply_species_xml(
                spec_xml,
                env_planting.id,
                "Plant trees: Mixed species environmental planting on land managed for environmental services",
                plant_date,
            )

        file_name = f"{layer}_{plot_idx:03d}"
        simulation.save_to_plo(f"{sim_path}/FullCAM_Plotfiles/{file_name}.plo")

        # Run the simulation
        simulation.run()
        simulation.save_csv(f"{sim_path}/FullCAM_Ouput/{file_name}.csv")
        df = simulation.to_dataframe()
        df["layer"] = layer
        df["idx"] = plot_idx
        df["area_ha"] = area

        all_results.append(df)

    # After the loop completes, concatenate all results
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(f"{sim_path}/FullCAM_Ouput/all_results.csv", index=False)
    combined_df.to_parquet(f"{sim_path}/FullCAM_Ouput/all_results.parquet", index=False)

    return combined_df


if __name__ == "__main__":
    # Initialize the FullCAM client
    client = FullCAMClient(version="2020")
    template = "ERF\\Environmental Plantings Method.plo"
    m_layer = "C:\\Development\\MullionGroup\\Wollemi-Demo\\FullCAM\\Input Data\\New_M_2019\\New_M_2019.tif"

    cea_files = [
        "C:\\Development\\MullionGroup\\Wollemi-Demo\\CEA-Stratification\\_Scenarios_15May_2025_Talbot_KangarooC\\KangarooCamp\\Scenario_Base\\Shapefiles\\KangarooCamp_plantable_area_mid.geojson",
        "C:\\Development\\MullionGroup\\Wollemi-Demo\\CEA-Stratification\\_Scenarios_15May_2025_Talbot_KangarooC\\KangarooCamp\\Scenario_Best\\Shapefiles\\KangarooCamp_plantable_area_best.geojson",
        "C:\\Development\\MullionGroup\\Wollemi-Demo\\CEA-Stratification\\_Scenarios_15May_2025_Talbot_KangarooC\\KangarooCamp\\Scenario_Worst\\Shapefiles\\KangarooCamp_plantable_area_worst.geojson",
        "C:\\Development\\MullionGroup\\Wollemi-Demo\\CEA-Stratification\\_Scenarios_15May_2025_Talbot_KangarooC\\Talbot\\Scenario_Base\\Shapefiles\\Talbot_plantable_area_mid.geojson",
        "C:\\Development\\MullionGroup\\Wollemi-Demo\\CEA-Stratification\\_Scenarios_15May_2025_Talbot_KangarooC\\Talbot\\Scenario_Best\\Shapefiles\\Talbot_plantable_area_best.geojson",
        "C:\\Development\\MullionGroup\\Wollemi-Demo\\CEA-Stratification\\_Scenarios_15May_2025_Talbot_KangarooC\\Talbot\\Scenario_Worst\\Shapefiles\\Talbot_plantable_area_worst.geojson",
    ]

    for file_path in cea_files:
        simulate_cea_cells(file_path, layer_property="CEA", m_layer=m_layer)
        # simulate_cea(file_path, layer_property="CEA", model_points_method="from_stats", method="less_than_mean")

    # accus=('C mass of trees  (tC/ha)'+'C mass of forest debris  (tC/ha)')*'area_ha'*44/12*0.95
    # combined_df['accus'] = (combined_df['C mass of trees  (tC/ha)'] + combined_df['C mass of forest debris  (tC/ha)']) * combined_df['area_ha'] * 44 / 12 * 0.95
    # combined_df.to_csv(f"{sim_path}\\all_results_accus.csv", index=False)
