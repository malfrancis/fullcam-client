"""
Simulation Manager for FullCAM client

This module provides classes for managing FullCAM plot simulations,
comparing their results, and creating visualizations.
"""

import io
import logging
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from xml.etree.ElementTree import Element

import pandas as pd
import pyarrow as pa
from pyarrow import csv

from fullcam_client.exceptions import FullCAMClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("fullcam_client.simulation")


@dataclass
class Build:
    def __init__(self, build: Element):
        self.latitude = build.get("latBL")
        self.longitude = build.get("lonBL")
        self.forest_category = build.get("frCat")

    latitude: float = 0.0
    longitude: float = 0.0
    forest_category: str = ""  # Forest category (e.g., "ERF", "NPP", etc.)


@dataclass
class Timing:
    def __init__(self, timing: Element):
        """
        <Timing dailyTimingTZ="true" useDaysPerStepDTZ="true" daysPerStepDTZ="1" stepsPerDayDTZ="1" outputFreqDTZ="Monthly" stepsPerOutDTZ="1" firstOutStepDTZ="1" tStepsYTZ="Monthly" stepsPerYrYTZ="110" stYrYTZ="" stStepInStYrYTZ="" enYrYTZ="" enStepInEnYrYTZ="" stepsPerOutYTZ="1" firstOutStepYTZ="1">
            <stDateTM CalendarSystemT="Gregorian">20100101</stDateTM>
            <enDateTM CalendarSystemT="Gregorian">21100101</enDateTM>
        </Timing>

        <Timing dailyTimingTZ="false" useDaysPerStepDTZ="true" daysPerStepDTZ="1" stepsPerDayDTZ="1" outputFreqDTZ="Daily" stepsPerOutDTZ="1" firstOutStepDTZ="1" tStepsYTZ="Monthly" stepsPerYrYTZ="110" stYrYTZ="2025" stStepInStYrYTZ="8" enYrYTZ="2050" enStepInEnYrYTZ="8" stepsPerOutYTZ="1" firstOutStepYTZ="1"/>

        """
        self.use_daily_timing = timing.get("dailyTimingTZ") == "true"
        if self.use_daily_timing:
            start_date_el = timing.find("stDateTM")
            end_date_el = timing.find("enDateTM")
            if start_date_el is not None:
                self.start_date = datetime.strptime(start_date_el.text, "%Y%m%d").date()
            if end_date_el is not None:
                self.end_date = datetime.strptime(end_date_el.text, "%Y%m%d").date()
        else:
            start_year = int(timing.get("stYrYTZ", "0"))
            start_step = int(timing.get("stStepInStYrYTZ", "0"))
            end_year = int(timing.get("enYrYTZ", "0"))
            end_step = int(timing.get("enStepInEnYrYTZ", "0"))
            self.start_date = datetime(start_year, 1, 1) + pd.DateOffset(months=start_step - 1)
            self.end_date = datetime(end_year, 1, 1) + pd.DateOffset(months=end_step - 1)

    start_date: datetime | None = None
    end_date: datetime | None = None


@dataclass
class About:
    def __init__(self, meta: Element):
        self.name = meta.get("nmME")
        notes_el = meta.find("notesME")
        if notes_el is not None:
            self.notes = notes_el.text
        else:
            self.notes = None
        self.version = meta.get("savedByVersion")

    name: str = ""
    notes: str | None = None
    version: str = ""


class Simulation:
    """
    Represents a single FullCAM plot simulation.

    This class handles:
    1. Loading a plot file
    2. Running the simulation
    3. Storing the results
    4. Converting the results to pandas DataFrame

    Attributes:
        name: Name of the simulation
        plot_file: Path to the plot file
        status: Status of the simulation (added, running, completed, failed)
        arrow_table: PyArrow table with the simulation results
        metadata: Dictionary with metadata about the simulation
    """

    def __init__(self, name: str, plot_file: str = None, xml_content: str = None, client=None):
        """
        Initialize a simulation.

        Args:
            name: Name for this simulation
            plot_file: Path to the plot file (.plo)
            xml_content: XML content as a string
            client: FullCAMClient instance (creates a new one if not provided)

        Raises:
            FileNotFoundError: If the plot file doesn't exist
            ValueError: If neither plot_file nor xml_content is provided
        """

        self.name = name
        from fullcam_client.client import FullCAMClient  # Import here to avoid circular import

        self.client = client if client is not None else FullCAMClient()
        self.arrow_table = None
        self.status = "added"
        self.metadata = {"added_time": datetime.now().isoformat(), "simulation_time": None}

        # Load XML content from file or directly
        if plot_file is not None:
            if not os.path.exists(plot_file):
                raise FileNotFoundError(f"Plot file not found: {plot_file}")
            self.plot_file = plot_file
            self.load_xml(plot_file)
            self.metadata["plot_file"] = plot_file
            logger.info(f"Initialized simulation '{name}' with plot file: {plot_file}")
        elif xml_content is not None:
            self.xml_content = xml_content
            logger.info(f"Initialized simulation '{name}' with provided XML content")
        else:
            raise ValueError("Either plot_file or xml_content must be provided")

        xml_stream = io.BytesIO(self.xml_content.encode())
        self.tree = ET.parse(xml_stream)
        root = self.tree.getroot()
        self.about = About(root.find("Meta"))
        self.timing = Timing(root.find("Timing"))
        self.build = Build(root.find("Build"))

    def load_xml(self, file_path: str) -> None:
        """
        Load XML content from a file

        Args:
            file_path: Path to the XML file

        Raises:
            FileNotFoundError: If the XML file is not found
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"XML file not found: {file_path}")

        with open(file_path, encoding="utf-8") as f:
            self.xml_content = f.read()

        logger.info(f"Loaded XML content from: {file_path}")

    def run(self) -> pa.Table:
        """
        Run the simulation and store the results.

        Returns:
            PyArrow table with simulation results

        Raises:
            FullCAMAPIError: If the API request fails
        """
        logger.info(f"Running simulation '{self.name}'...")
        self.status = "running"
        start_time = datetime.now()

        try:
            # Run the simulation
            self.arrow_table = self.client.simulate(self.xml_content)

            # Update metadata
            end_time = datetime.now()
            self.status = "completed"
            self.metadata["simulation_time"] = end_time.isoformat()
            self.metadata["duration_seconds"] = (end_time - start_time).total_seconds()

            logger.info(f"Simulation '{self.name}' completed with {len(self.arrow_table)} rows")
            return self.arrow_table

        except Exception as e:
            self.status = "failed"
            self.metadata["error"] = str(e)
            logger.error(f"Simulation '{self.name}' failed: {e}")
            raise

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the simulation results to a pandas DataFrame.

        Returns:
            pandas DataFrame

        Raises:
            ValueError: If the simulation hasn't been run
        """
        if self.status != "completed":
            raise ValueError(f"Simulation '{self.name}' hasn't been run successfully.")

        if self.arrow_table is None:
            raise ValueError(f"Simulation '{self.name}' has no results.")

        df = self.arrow_table.to_pandas()

        return df

    def to_csv(self) -> str:
        """
        Convert the simulation results to CSV.

        Returns:
            CSV data as a string

        Raises:
            ValueError: If the simulation hasn't been run
        """
        if self.status != "completed":
            raise ValueError(f"Simulation '{self.name}' hasn't been run successfully.")

        if self.arrow_table is None:
            raise ValueError(f"Simulation '{self.name}' has no results.")
        try:
            # Convert the arrow table to CSV
            csv_buffer = io.BytesIO()
            csv.write_csv(self.arrow_table, csv_buffer)
            csv_data = csv_buffer.getvalue().decode("utf-8")
            logger.info(f"Converted Arrow table to CSV with {csv_data.count(chr(10)) + 1} lines")
            return csv_data
        except Exception as e:
            logger.error(f"Failed to convert Arrow table to CSV: {str(e)}")
            raise FullCAMClientError(f"Failed to convert Arrow table to CSV: {str(e)}") from e

    def save_csv(self, file_path: str) -> None:
        """
        Save the simulation results to a CSV file.

        Args:
            file_path: Path to save the CSV file

        Raises:
            ValueError: If the simulation hasn't been run
        """

        csv_data = self.to_csv()

        try:
            with open(file_path, "w", encoding="utf-8", newline="") as f:
                f.write(csv_data)

            logger.info(f"Saved CSV data to: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save CSV to file: {str(e)}")
            raise FullCAMClientError(f"Failed to save CSV to file: {str(e)}") from e

        logger.info(f"Saved simulation '{self.name}' results to {file_path}")

    def get_metadata(self) -> dict:
        """
        Export metadata about the simulation.

        Returns:
            Dictionary with metadata
        """
        return {
            "name": self.name,
            "plot_file": getattr(self, "plot_file", None),
            "status": self.status,
            "row_count": len(self.arrow_table) if self.arrow_table is not None else None,
            "metadata": self.metadata,
        }

    def update_time_series(self, old_ts: Element, new_ts: Element) -> None:
        try:
            old_ts.clear()
            for attr in new_ts.attrib:
                old_ts.set(attr, new_ts.get(attr))
            for child in new_ts:
                old_ts.append(child)
        except Exception as e:
            logger.error(f"Failed to update time series data: {str(e)}")
            raise FullCAMClientError(f"Failed to update time series data: {str(e)}") from e

    def apply_location_xml(self, xml_content: str) -> None:
        """
        Apply location XML content to the simulation.

        Args:
            xml_content: XML content as a string

        Raises:
            ValueError: If the XML content is invalid
        """
        try:
            doc_root = self.tree.getroot()

            xml_stream = io.BytesIO(xml_content.encode())
            location_tree = ET.parse(xml_stream)
            location_root = location_tree.getroot()

            site_info = location_root.find("SiteInfo")
            site = doc_root.find("Site")
            if site_info is not None:
                logger.info(
                    f"Location data downloaded for {site_info.get('state')} SA2 {site_info.get('sa2Name')}(siteId: {site_info.get('SA2')}) NPI {site_info.get('npi')} "
                )

                new_soil_base = location_root.find("LocnSoil/SoilBase")
                soil_base = doc_root.find("Soil/SoilBase")
                soil_base.clear()
                for child in new_soil_base:
                    soil_base.append(child)

                maxAbgMF = location_root.find("InputElement[@tIn='maxAbgMF']").get("value")
                fpiAvgLT = location_root.find("InputElement[@tIn='fpiAvgLT']").get("value")
                site.set("maxAbgMF", maxAbgMF)
                site.set("fpiAvgLT", fpiAvgLT)

                new_rainfall = location_root.find("InputElement[@tIn='rainfall']").find(
                    "TimeSeries"
                )
                if new_rainfall is not None:
                    rainfall = site.find("TimeSeries[@tInTS='rainfall']")
                    self.update_time_series(rainfall, new_rainfall)

                new_openPanEvap = location_root.find("InputElement[@tIn='openPanEvap']").find(
                    "TimeSeries"
                )
                if new_openPanEvap is not None:
                    openPanEvap = site.find("TimeSeries[@tInTS='openPanEvap']")
                    self.update_time_series(openPanEvap, new_openPanEvap)

                new_avgAirTemp = location_root.find("InputElement[@tIn='avgAirTemp']").find(
                    "TimeSeries"
                )
                if new_avgAirTemp is not None:
                    avgAirTemp = site.find("TimeSeries[@tInTS='avgAirTemp']")
                    self.update_time_series(avgAirTemp, new_avgAirTemp)

                tAirTemp = location_root.find("InputElement[@tIn='tAirTemp']").get("value")
                new_forestProdIx = location_root.find("InputElement[@tIn='forestProdIx']").find(
                    "TimeSeries"
                )
                if new_forestProdIx is not None:
                    forestProdIx = site.find("TimeSeries[@tInTS='forestProdIx']")
                    self.update_time_series(forestProdIx, new_forestProdIx)

                location_soil = location_root.find("LocnSoil")
                initFracDpma = float(location_soil.get("initFracDpma"))
                initFracRpma = float(location_soil.get("initFracRpma"))
                initFracHums = float(location_soil.get("initFracHums"))
                initFracInrt = float(location_soil.get("initFracInrt"))
                initFracBiof = float(location_soil.get("initFracBiof"))
                initFracBios = float(location_soil.get("initFracBios"))
                initTotalC = float(location_soil.get("initTotalC"))
                soilCoverA = float(location_soil.get("soilCoverA"))

                init_soil_f = doc_root.find("Init/InitSoilF")
                init_soil_f.set("dpmaCMInitF", str(initTotalC * initFracDpma))
                init_soil_f.set("rpmaCMInitF", str(initTotalC * initFracRpma))
                init_soil_f.set("humsCMInitF", str(initTotalC * initFracHums))
                init_soil_f.set("inrtCMInitF", str(initTotalC * initFracInrt))
                init_soil_f.set("biofCMInitF", str(initTotalC * initFracBiof))
                init_soil_f.set("biosCMInitF", str(initTotalC * initFracBios))

                init_soil_a = doc_root.find("Init/InitSoilA")
                init_soil_a.set("dpmaCMInitA", str(0.0))
                init_soil_a.set("rpmaCMInitA", str(0.0))
                init_soil_a.set("humsCMInitA", str(0.0))
                init_soil_a.set("inrtCMInitA", str(0.0))
                init_soil_a.set("biofCMInitA", str(0.0))
                init_soil_a.set("biosCMInitA", str(0.0))

                forest_species_list = []
                forest_species = location_root.find("ItemList[@id='FrSpecies']")
                for item in forest_species:
                    spec={
                        "id": item.get("id"),
                        "name": item.get("value"),
                    }
                    forest_species_list.append(spec)

                


            logger.info(f"Applied location XML content to simulation '{self.name}'")

            return forest_species_list
        except Exception as e:
            logger.error(f"Failed to apply location XML: {str(e)}")
            raise FullCAMClientError(f"Failed to apply location XML: {str(e)}") from e

    def save_to_plo(self, file_path: str) -> None:
        """
        Save the simulation XML content to a .plo file.

        Args:
            file_path: Path to save the .plo file

        Raises:
            ValueError: If the simulation doesn't have XML content
        """
        if not self.tree:
            raise ValueError(f"Simulation '{self.name}' doesn't have XML content.")

        try:
            root = self.tree.getroot()

            # set the page to the about tab
            root.set("pageIxDO", "0")
            meta = root.find("Meta")
            if meta is not None:
                meta.set("nmME", self.about.name)
                notes_el = meta.find("notesME")
                if notes_el is not None:
                    notes_el.text = self.about.notes
                else:
                    logger.warning("Notes element not found in XML. Skipping notes update.")
            else:
                logger.warning("Meta element not found in XML. Skipping name and notes update.")

            timing = root.find("Timing")
            if timing is not None:
                start_date_el = timing.find("stDateTM")
                end_date_el = timing.find("enDateTM")
                if self.timing.use_daily_timing:
                    timing.set("dailyTimingTZ", "true")
                    timing.set("stYrYTZ", "")
                    timing.set("stStepInStYrYTZ", "")
                    timing.set("enYrYTZ", "")
                    timing.set("enStepInEnYrYTZ", "")
                    if start_date_el is None:
                        start_date_el = ET.SubElement(timing, "stDateTM")
                    if end_date_el is None:
                        end_date_el = ET.SubElement(timing, "enDateTM")
                    if self.timing.start_date:
                        start_date_el.text = self.timing.start_date.strftime("%Y%m%d")
                    if self.timing.end_date:
                        end_date_el.text = self.timing.end_date.strftime("%Y%m%d")
                else:
                    start_year = self.timing.start_date.year
                    start_step = self.timing.start_date.month
                    end_year = self.timing.end_date.year
                    end_step = self.timing.end_date.month

                    timing.set("dailyTimingTZ", "false")
                    timing.set("stYrYTZ", str(start_year))
                    timing.set("stStepInStYrYTZ", str(start_step))
                    timing.set("enYrYTZ", str(end_year))
                    timing.set("enStepInEnYrYTZ", str(end_step))
                    timing.remove(start_date_el)
                    timing.remove(end_date_el)

            build = root.find("Build")
            if build is not None:
                build.set("latBL", str(self.build.latitude))
                build.set("lonBL", str(self.build.longitude))
                build.set("frCat", self.build.forest_category)
            else:
                logger.warning(
                    "Build element not found in XML. Skipping latitude and longitude update."
                )

            ET.indent(self.tree)
            self.tree.write(file_path, encoding="utf-8", xml_declaration=True, method="xml")

            # Update plot_file attribute
            self.plot_file = file_path
            self.metadata["plot_file"] = file_path
            logger.info(f"Saved simulation '{self.name}' to plo file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save plo file: {str(e)}")
            raise FullCAMClientError(f"Failed to save plo file: {str(e)}") from e
