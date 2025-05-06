from fullcam_client.client import FullCAMClient
from fullcam_client.simulation import Simulation
import geopandas as gpd
import pandas as pd

if __name__ == "__main__":

    # Initialize the FullCAM client
    client = FullCAMClient(version="2020")
    template = 'ERF\\Environmental Plantings Method.plo'

    gdf = gpd.read_file("examples/BelokaCEAsMerged.geojson")
    #gdf.to_crs(9473)
    #print(gdf.crs)
    for i in range(len(gdf)):
        centroid = gdf.at[i, 'geometry'].centroid 

        area = gdf.at[i, 'geometry'].area*(10**6) # Convert to hectares
        area2 = gdf.at[i, 'Area (ha)']

        projected_gdf = gdf.to_crs(epsg=28355)  # MGA Zone 55 (for eastern Australia)
        # Calculate area in square meters
        area_m2 = projected_gdf.geometry.area
        # Convert to hectares (1 ha = 10,000 m²)
        area_ha = area_m2 / 10000        

        projected_gdf = gdf.to_crs(epsg=3577)  # Australian Albers Equal Area
        area_m2 = projected_gdf.geometry.area

        plant_date = gdf.at[i, 'Plant Date']
        config = gdf.at[i, 'Configuration']
        layer = gdf.at[i, 'layer']

        simulation = client.create_simulation_from_template(template, layer)
        simulation.about.name = layer
        simulation.about.notes = f"Simulation for {layer}\n\nArea: {area} ha\n\nPlant Date: {plant_date}\n\nConfiguration: {config}"
        simulation.build.latitude = centroid.y
        simulation.build.longitude = centroid.x
        simulation.timing.start_date = plant_date
        simulation.timing.end_date = plant_date + pd.DateOffset(years=25)
        simulation.save_to_plo(f"examples/{layer}.plo")
        #update the site data

        #t = simulation.run()
        #print(t)
