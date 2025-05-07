import json
from datetime import date, datetime

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

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


if __name__ == "__main__":
    # Initialize the FullCAM client
    client = FullCAMClient(version="2020")
    template = "ERF\\Environmental Plantings Method.plo"

    gdf = gpd.read_file("examples/BelokaCEAsMerged.geojson")
    gdf["centroid"] = gdf.geometry.centroid

    projected_gdf = gdf.to_crs(epsg=3577)  # Australian Albers Equal Area
    projected_gdf["centroid"] = projected_gdf.geometry.centroid
    projected_gdf["area_ha_albers"] = projected_gdf.geometry.area / 10000

    for i in range(len(gdf)):
        centroid = gdf.at[i, "geometry"].centroid
        area = projected_gdf.at[i, "area_ha_albers"]

        plant_date = gdf.at[i, "Plant Date"]
        config = gdf.at[i, "Configuration"]
        layer = gdf.at[i, "layer"]

        simulation = client.create_simulation_from_template(template, layer)

        notes = {
            "layer": layer,
            "area": area,
            "plant_date": plant_date.strftime("%Y-%m-%d"),
            "centroid": {"latitude": centroid.y, "longitude": centroid.x},
            "configuration": config,
            "properties": convert_for_json(gdf.iloc[i].drop("geometry").to_dict()),
        }
        simulation.about.name = layer
        simulation.about.notes = json.dumps(notes)
        simulation.timing.use_daily_timing = False
        simulation.timing.start_date = plant_date
        simulation.timing.end_date = plant_date + pd.DateOffset(years=25)
        simulation.build.latitude = centroid.y
        simulation.build.longitude = centroid.x
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


        simulation.save_to_plo(f"examples/{layer}.plo")

        # Update the site data and get the species

        # Add the Plant Event

        # Run the simulation
        results = simulation.run()
        simulation.save_csv(f"examples/{layer}.csv")
