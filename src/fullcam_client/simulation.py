"""
Simulation Manager for FullCAM client

This module provides classes for managing FullCAM plot simulations,
comparing their results, and creating visualizations.
"""

import copy
import io
import logging
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from xml.etree.ElementTree import Element

import pandas as pd
import pyarrow as pa
from blinker import signal
from pyarrow import csv
from pydantic import BaseModel, Field

from fullcam_client.exceptions import FullCAMClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("fullcam_client.simulation")


class Build(BaseModel):
    latitude: float = 0.0
    longitude: float = 0.0
    forest_category: str = ""  # Forest category (e.g., "ERF", "NPP", etc.)

    def __init__(self, build=None, **data):
        if build is not None:
            data["latitude"] = float(build.get("latBL", 0.0))
            data["longitude"] = float(build.get("lonBL", 0.0))
            data["forest_category"] = build.get("frCat", "")
        super().__init__(**data)

    def model_post_init(self, __context):
        """Hook called after the model is fully initialized"""
        # Store original values for change detection
        self._original_values = dict(self)

    def __setattr__(self, name, value):
        """Override setattr to detect property changes"""
        if name in self.model_fields_set:
            old_value = getattr(self, name) if hasattr(self, name) else None

            # Only emit signal if the value is actually changing
            if old_value != value:
                # Set the new value
                super().__setattr__(name, value)

                # Emit the signal
                build_changed = signal("build_changed")
                build_changed.send(self, field_name=name, old_value=old_value, new_value=value)
            else:
                super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)


class Timing(BaseModel):
    start_date: datetime | None = None
    end_date: datetime | None = None
    use_daily_timing: bool = False

    def __init__(self, timing=None, **data):
        if timing is not None:
            use_daily_timing = str(timing.get("dailyTimingTZ")).lower() == "true"
            data["use_daily_timing"] = use_daily_timing
            if use_daily_timing:
                start_date_el = timing.find("stDateTM")
                end_date_el = timing.find("enDateTM")
                if start_date_el is not None:
                    data["start_date"] = datetime.strptime(start_date_el.text, "%Y%m%d").date()
                if end_date_el is not None:
                    data["end_date"] = datetime.strptime(end_date_el.text, "%Y%m%d").date()
            else:
                start_year = int(timing.get("stYrYTZ", "0"))
                start_step = int(timing.get("stStepInStYrYTZ", "0"))
                end_year = int(timing.get("enYrYTZ", "0"))
                end_step = int(timing.get("enStepInEnYrYTZ", "0"))
                data["start_date"] = datetime(start_year, 1, 1) + pd.DateOffset(
                    months=start_step - 1
                )
                data["end_date"] = datetime(end_year, 1, 1) + pd.DateOffset(months=end_step - 1)
        super().__init__(**data)

    def model_post_init(self, __context):
        """Hook called after the model is fully initialized"""
        # Store original values for change detection
        self._original_values = dict(self)

    def __setattr__(self, name, value):
        """Override setattr to detect property changes"""
        if name in self.model_fields_set:
            old_value = getattr(self, name) if hasattr(self, name) else None

            # Only emit signal if the value is actually changing
            if old_value != value:
                # Set the new value
                super().__setattr__(name, value)

                # Emit the signal
                timing_changed = signal("timing_changed")
                timing_changed.send(self, field_name=name, old_value=old_value, new_value=value)
            else:
                super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)


class About(BaseModel):
    name: str = ""
    notes: str | None = None
    version: str = ""

    def __init__(self, meta=None, **data):
        if meta is not None:
            data["name"] = meta.get("nmME")
            data["notes"] = meta.get("notesME")
            data["version"] = meta.get("savedByVersion")
        super().__init__(**data)

    def model_post_init(self, __context):
        """Hook called after the model is fully initialized"""
        # Store original values for change detection
        self._original_values = dict(self)

    def __setattr__(self, name, value):
        """Override setattr to detect property changes"""
        if name in self.model_fields_set:
            old_value = getattr(self, name) if hasattr(self, name) else None

            # Only emit signal if the value is actually changing
            if old_value != value:
                # Set the new value
                super().__setattr__(name, value)

                # Emit the signal
                about_changed = signal("about_changed")
                about_changed.send(self, field_name=name, old_value=old_value, new_value=value)
            else:
                super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)


class LocationSpecies(BaseModel):
    id: str = ""
    name: str = ""
    tyf_parameters: list[dict] = Field(default_factory=list)

    def __init__(self, species=None, **data):
        if species is not None:
            data["id"] = species.get("id")
            data["name"] = species.get("value")
            # Example XML structure for TYFParameters

            """
                <TYFParameters count="4" idSP="35">
                    <TYFCategory tTYFCat="BeltH" tyf_G="3.492" tyf_r="1.2" />
                    <TYFCategory tTYFCat="BeltL" tyf_G="4.533" tyf_r="1.2" />
                    <TYFCategory tTYFCat="BlockES" tyf_G="6.317" tyf_r="1.0" />
                    <TYFCategory tTYFCat="Water" tyf_G="5.724" tyf_r="1.2" />
                </TYFParameters>
            """
            # Extract TYFParameters if present
            tyf_params = species.find("TYFParameters")
            if tyf_params is not None:
                data["tyf_parameters"] = []
                for tyf_category in tyf_params:
                    category_data = {
                        "category": tyf_category.get("tTYFCat"),
                        "tyf_G": float(tyf_category.get("tyf_G", 0.0)),
                        "tyf_r": float(tyf_category.get("tyf_r", 0.0)),
                    }
                    data["tyf_parameters"].append(category_data)

        super().__init__(**data)


class LocationInfo(BaseModel):
    state: str = ""
    SA2_name: str | None = None
    SA2: str = ""
    npi: str = ""
    maximum_aboveground_biomass: float = 0.0
    long_term_average_FPI: float = 0.0
    forest_species: list[LocationSpecies] = Field(default_factory=list)
    location_soil: dict = Field(default_factory=dict)

    def __init__(self, location_root=None, **data):
        site_info = location_root.find("SiteInfo")
        if site_info is not None:
            data["state"] = site_info.get("state")
            data["SA2_name"] = site_info.get("sa2Name")
            data["SA2"] = site_info.get("SA2")
            data["npi"] = site_info.get("npi")

        data_location_soil = {}
        location_soil = location_root.find("LocnSoil")


        if location_soil is not None:
            data_location_soil["initFracDpma"] = float(location_soil.get("initFracDpma"))
            data_location_soil["initFracRpma"] = float(location_soil.get("initFracRpma"))
            data_location_soil["initFracBiof"] = float(location_soil.get("initFracBiof"))
            data_location_soil["initFracBios"] = float(location_soil.get("initFracBios"))
            data_location_soil["initFracHums"] = float(location_soil.get("initFracHums"))
            data_location_soil["initFracInrt"] = float(location_soil.get("initFracInrt"))
            data_location_soil["initTotalC"] = float(location_soil.get("initTotalC"))
            data_location_soil["soilCoverA"] = float(location_soil.get("soilCoverA"))

            init_TSMD = location_soil.find("TimeSeries[@tInTS='initTSMD']")
            if init_TSMD is not None:
                data_location_soil["init_TSMD"] = {
                    "tExtrapTS": init_TSMD.get("tExtrapTS"),
                    "tOriginTS": init_TSMD.get("tOriginTS"),
                    "yr0TS": int(init_TSMD.get("yr0TS", 0)),
                    "nYrsTS": int(init_TSMD.get("nYrsTS", 0)),
                    "dataPerYrTS": int(init_TSMD.get("dataPerYrTS", 1)),
                }
                raw_ts = init_TSMD.find("rawTS")
                if raw_ts is not None:
                    data_location_soil["init_TSMD"]["raw_values"] = [
                        float(i) for i in raw_ts.text.strip().split(",")
                    ]


            soil_base = location_soil.find("SoilBase")
            """
                <SoilZap id="forest" maxBiofNCRatio="" maxBiosNCRatio="" maxHumsNCRatio="" minBiofNCRatio="" minBiosNCRatio="" minHumsNCRatio="" fracHumfSopmToDpma="0.9" fracHumfLrpmToRpma="0.9" fracHumfMrpmToRpma="0.9" fracHumfSommToDpma="0.9" fracHumfLrmmToRpma="0.9" fracHumfMrmmToRpma="0.9" fracHumfMicrToXpma="0.9" fracDLitBkdnToDpma="1.0" fracRLitBkdnToRpma="1.0" dToRRatioInPres="0.2" doManuFromOffs="false" fracManuCMToDpma="0.49" fracManuCMToRpma="0.49" fracManuCMToBiof="0.0" fracManuCMToBios="0.0" manuDpmaNCRatio="100.0" manuRpmaNCRatio="100.0" manuBiofNCRatio="100.0" manuBiosNCRatio="100.0" manuHumsNCRatio="100.0" encpFracHums="0.0" sampleDepth="30.0" pH="6.0" evapoOpenRatio="0.75" bToCMaxTSMDRatio="0.556" sdcmRateMultDpma="10.0" sdcmRateMultRpma="0.17" sdcmRateMultBiofV263="0.66" sdcmRateMultBiosV263="0.66" sdcmRateMultHums="0.03" sdcmRateMultBiomCov="3.25" fracPbioToBiofV263="0.46" fracPbioToBiofV265="0.28" fracPbioToBiosV265="0.18" fracHumsToBiosV263="0.46" fracHumsToBiofV265="0.28" fracHumsToBiosV265="0.18" richNCRatio="0.01" poorNCRatio="0.001" fracDpmaToStorMyco="0.0" fracRpmaToStorMyco="0.0" fracBiofToStorMyco="0.0" fracBiosToStorMyco="0.0" />
                <SoilZap id="agriculture" maxBiofNCRatio="" maxBiosNCRatio="" maxHumsNCRatio="" minBiofNCRatio="" minBiosNCRatio="" minHumsNCRatio="" fracHumfSopmToDpma="0.9" fracHumfLrpmToRpma="0.9" fracHumfMrpmToRpma="0.9" fracHumfSommToDpma="0.9" fracHumfLrmmToRpma="0.9" fracHumfMrmmToRpma="0.9" fracHumfMicrToXpma="0.9" fracDLitBkdnToDpma="1.0" fracRLitBkdnToRpma="1.0" dToRRatioInPres="0.2" doManuFromOffs="false" fracManuCMToDpma="0.49" fracManuCMToRpma="0.49" fracManuCMToBiof="0.0" fracManuCMToBios="0.0" manuDpmaNCRatio="100.0" manuRpmaNCRatio="100.0" manuBiofNCRatio="100.0" manuBiosNCRatio="100.0" manuHumsNCRatio="100.0" encpFracHums="0.0" sampleDepth="30.0" pH="6.0" evapoOpenRatio="0.75" bToCMaxTSMDRatio="0.556" sdcmRateMultDpma="10.0" sdcmRateMultRpma="0.17" sdcmRateMultBiofV263="0.66" sdcmRateMultBiosV263="0.66" sdcmRateMultHums="0.03" sdcmRateMultBiomCov="3.25" fracPbioToBiofV263="0.46" fracPbioToBiofV265="0.28" fracPbioToBiosV265="0.18" fracHumsToBiosV263="0.46" fracHumsToBiofV265="0.28" fracHumsToBiosV265="0.18" richNCRatio="0.01" poorNCRatio="0.001" fracDpmaToStorMyco="0.0" fracRpmaToStorMyco="0.0" fracBiofToStorMyco="0.0" fracBiosToStorMyco="0.0" />
                <SoilOther id="other" clayFrac="0.1609364897013" bulkDensity="1.42" tSoil="ClayLoam" maxASW="300.0" />
            """
            if soil_base is not None:
                # Extract soil base properties
                soil_forest = soil_base.find("SoilZap[@id='forest']")
                soil_agriculture = soil_base.find("SoilZap[@id='agriculture']")
                soil_other = soil_base.find("SoilOther[@id='other']")

            data["location_soil"] = data_location_soil

        maxAbgMF = location_root.find("InputElement[@tIn='maxAbgMF']").get("value")
        if maxAbgMF is not None:
            data["maximum_aboveground_biomass"] = maxAbgMF
        fpiAvgLT = location_root.find("InputElement[@tIn='fpiAvgLT']").get("value")
        if fpiAvgLT is not None:
            data["long_term_average_FPI"] = fpiAvgLT

        item_list = location_root.find("ItemList[@id='FrSpecies']")
        if item_list is not None:
            data["forest_species"] = []
            for item in item_list:
                # Create a new LocationSpecies instance for each item
                spec = LocationSpecies(item)
                data["forest_species"].append(spec)

        super().__init__(**data)

    def model_post_init(self, __context):
        """Hook called after the model is fully initialized"""
        # Store original values for change detection
        self._original_values = dict(self)

    def __setattr__(self, name, value):
        """Override setattr to detect property changes"""
        if name in self.model_fields_set:
            old_value = getattr(self, name) if hasattr(self, name) else None

            # Only emit signal if the value is actually changing
            if old_value != value:
                # Set the new value
                super().__setattr__(name, value)

                # Emit the signal
                about_changed = signal("about_changed")
                about_changed.send(self, field_name=name, old_value=old_value, new_value=value)
            else:
                super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)


class Simulation:
    doc_tree: ET.ElementTree | None = None
    location_info_tree: ET.ElementTree | None = None

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
        self.doc_tree = ET.parse(xml_stream)
        root = self.doc_tree.getroot()
        self.about = About(root.find("Meta"))
        self.timing = Timing(root.find("Timing"))
        self.build = Build(root.find("Build"))
        self.location_info = None

        # Connect signals to handlers
        signal("about_changed").connect(self._on_about_changed)
        signal("timing_changed").connect(self._on_timing_changed)
        signal("build_changed").connect(self._on_build_changed)

    def _on_timing_changed(self, sender, field_name, old_value, new_value):
        """Handler for timing property changes"""
        logger.info(f"Timing property '{field_name}' changed from {old_value} to {new_value}")

        # Update XML tree
        root = self.doc_tree.getroot()
        timing_el = root.find("Timing")

        if timing_el is not None:
            if field_name == "start_date":
                if self.timing.use_daily_timing:
                    start_date_el = timing_el.find("stDateTM")
                    start_date_el.text = new_value.strftime("%Y%m%d")
                else:
                    timing_el.set("stYrYTZ", str(new_value.year))
                    timing_el.set("stStepInStYrYTZ", str(new_value.month))

            elif field_name == "end_date":
                if self.timing.use_daily_timing:
                    end_date_el = timing_el.find("enDateTM")
                    end_date_el.text = new_value.strftime("%Y%m%d")
                else:
                    timing_el.set("enYrYTZ", str(new_value.year))
                    timing_el.set("enStepInEnYrYTZ", str(new_value.month))

            elif field_name == "use_daily_timing":
                timing_el.set("dailyTimingTZ", str(new_value).lower())
                if new_value:  # daily timing
                    start_date_el = ET.SubElement(timing_el, "stDateTM")
                    start_date_el = ET.SubElement(timing_el, "enDateTM")
                    if self.timing.start_date:
                        start_date_el.text = self.timing.start_date.strftime("%Y%m%d")
                    if self.timing.end_date:
                        end_date_el.text = self.timing.end_date.strftime("%Y%m%d")
                else:  # step timing
                    start_year = self.timing.start_date.year
                    start_step = self.timing.start_date.month
                    end_year = self.timing.end_date.year
                    end_step = self.timing.end_date.month

                    timing_el.set("stYrYTZ", str(start_year))
                    timing_el.set("stStepInStYrYTZ", str(start_step))
                    timing_el.set("enYrYTZ", str(end_year))
                    timing_el.set("enStepInEnYrYTZ", str(end_step))
                    start_date_el = timing_el.find("stDateTM")
                    end_date_el = timing_el.find("enDateTM")
                    if start_date_el is not None:
                        timing_el.remove(start_date_el)
                    if end_date_el is not None:
                        timing_el.remove(end_date_el)

    def _on_build_changed(self, sender, field_name, old_value, new_value):
        """Handler for build property changes"""
        logger.info(f"Build property '{field_name}' changed from {old_value} to {new_value}")

        # Update XML tree
        root = self.doc_tree.getroot()
        build_el = root.find("Build")

        if build_el is not None:
            if field_name == "latitude":
                build_el.set("latBL", str(new_value))
            elif field_name == "longitude":
                build_el.set("lonBL", str(new_value))
            elif field_name == "forest_category":
                build_el.set("frCat", new_value)

    def _on_about_changed(self, sender, field_name, old_value, new_value):
        """Handler for metadata property changes"""
        logger.info(f"Metadata property '{field_name}' changed from {old_value} to {new_value}")

        # Update XML tree
        root = self.doc_tree.getroot()
        meta_el = root.find("Meta")

        if meta_el is not None:
            if field_name == "name":
                meta_el.set("nmME", new_value)
            elif field_name == "notes":
                notes_el = meta_el.find("notesME")
                if notes_el is not None:
                    notes_el.text = new_value
                else:
                    logger.warning("Notes element not found in XML. Skipping notes update.")

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
            root = self.doc_tree.getroot()
            plot_xml = ET.tostring(
                root,
                encoding="utf-8",
                method="xml",
                xml_declaration=True,
                short_empty_elements=True,
            ).decode("utf-8")

            # Run the simulation
            self.arrow_table = self.client.simulate(plot_xml)

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

    def get_time_series_value(self, time_series_element, target_date=None):
        input_id = time_series_element.get("tInTS")
        extrap_type = time_series_element.get("tExtrapTS")
        origin = time_series_element.get("tOriginTS")
        start_year = int(time_series_element.get("yr0TS", "0"))
        data_per_year = int(time_series_element.get("dataPerYrTS", "1"))
        if extrap_type != "AvgYr" or data_per_year != 1 or origin != "Calendar":
            raise ValueError(
                f"Unsupported TimeSeries ({input_id}) attributes: tExtrapTS={extrap_type}, dataPerYrTS={data_per_year}, tOriginTS={origin}"
            )

        # Get the raw data
        rawTS = time_series_element.find("rawTS")
        if rawTS is None or not rawTS.text:
            return 0.0

        # Parse raw values
        raw = rawTS.text.strip().split(",")
        raw_values = [float(i) for i in raw]

        if not raw_values:
            return 0.0

        # Calculate average for default/fallback
        average_value = sum(raw_values) / len(raw_values)

        # If no target date provided, return the average
        if target_date is None:
            return average_value

        try:
            # Calculate end year
            total_years = len(raw_values) / data_per_year
            end_year = start_year + total_years

            # Get target year
            target_year = target_date.year

            # Check if date is within range
            if target_year < start_year or target_year >= end_year:
                return average_value

            # Calculate index
            years_offset = target_year - start_year
            index = int(years_offset * data_per_year + 0.0000001)  # Avoid floating point issues

            # For multiple data points per year, adjust index based on date position in year
            if data_per_year > 1:
                start_of_year = datetime(target_year, 1, 1)
                if isinstance(target_date, datetime):
                    year_fraction = (target_date - start_of_year).total_seconds() / (
                        365.25 * 24 * 3600
                    )
                else:  # Assume it's a date
                    year_fraction = (target_date - start_of_year.date()).days / 365.25

                index += int(
                    year_fraction * data_per_year + 0.0000001
                )  # Avoid floating point issues

            # Ensure index is within bounds
            index = min(max(0, index), len(raw_values) - 1)

            return raw_values[index]

        except (ValueError, TypeError, AttributeError):
            # If any calculation fails, return the average
            return average_value

    def download_location_info(self, apply_queried_data: bool = True) -> None:
        xml = self.client.get_location_xml(
            self.build.latitude,
            self.build.longitude,
            forest_category=self.build.forest_category,
        )
        if xml is None:
            raise FullCAMClientError(
                f"Failed to get location XML for latitude {self.build.latitude} and longitude {self.build.longitude}"
            )
        xml_stream = io.BytesIO(xml.encode())
        self.location_info_tree = ET.parse(xml_stream)
        location_root = self.location_info_tree.getroot()
        self.location_info = LocationInfo(location_root)
        if apply_queried_data:
            self.apply_location_xml()

    def apply_location_xml(self) -> None:
        """
        Apply location XML content to the simulation.

        Args:
            xml_content: XML content as a string

        Raises:
            ValueError: If the XML content is invalid
        """
        try:
            doc_root = self.doc_tree.getroot()
            location_root = self.location_info_tree.getroot()

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

                if location_soil is not None:
                    TSMDInitF = 0.0
                    initTSMD = location_soil.find("TimeSeries[@tInTS='initTSMD']")
                    if initTSMD is not None:
                        TSMDInitF = self.get_time_series_value(initTSMD, self.timing.start_date)

                    initFracDpma = float(location_soil.get("initFracDpma"))
                    initFracRpma = float(location_soil.get("initFracRpma"))
                    initFracHums = float(location_soil.get("initFracHums"))
                    initFracInrt = float(location_soil.get("initFracInrt"))
                    initFracBiof = float(location_soil.get("initFracBiof"))
                    initFracBios = float(location_soil.get("initFracBios"))
                    initTotalC = float(location_soil.get("initTotalC"))

                    # not sure what this is for
                    # maybe set TimeSeries tInTS="soilCover" ??
                    soilCoverA = float(location_soil.get("soilCoverA"))

                    init_soil_f = doc_root.find("Init/InitSoilF")
                    init_soil_f.set("dpmaCMInitF", str(initTotalC * initFracDpma))
                    init_soil_f.set("rpmaCMInitF", str(initTotalC * initFracRpma))
                    init_soil_f.set("humsCMInitF", str(initTotalC * initFracHums))
                    init_soil_f.set("inrtCMInitF", str(initTotalC * initFracInrt))
                    init_soil_f.set("biofCMInitF", str(initTotalC * initFracBiof))
                    init_soil_f.set("biosCMInitF", str(initTotalC * initFracBios))
                    init_soil_f.set("TSMDInitF", str(TSMDInitF))

                    init_soil_a = doc_root.find("Init/InitSoilA")
                    init_soil_a.set("dpmaCMInitA", str(0.0))
                    init_soil_a.set("rpmaCMInitA", str(0.0))
                    init_soil_a.set("humsCMInitA", str(0.0))
                    init_soil_a.set("inrtCMInitA", str(0.0))
                    init_soil_a.set("biofCMInitA", str(0.0))
                    init_soil_a.set("biosCMInitA", str(0.0))
                    init_soil_a.set("TSMDInitA", "")

            logger.info(f"Applied location XML content to simulation '{self.name}'")

        except Exception as e:
            logger.error(f"Failed to apply location XML: {str(e)}")
            raise FullCAMClientError(f"Failed to apply location XML: {str(e)}") from e

    def apply_species_xml(self, spec_xml, species_id, plant_event_name, plant_date) -> None:
        """
        Apply species XML content to the simulation.

        Args:
            spec_xml: XML content as a string
            species_id: Species ID to apply

        Raises:
            ValueError: If the XML content is invalid
        """
        try:
            doc_root = self.doc_tree.getroot()
            doc_species_forest_set = doc_root.find("SpeciesForestSet")

            xml_stream = io.BytesIO(spec_xml.encode())
            xml_species_tree = ET.parse(xml_stream)
            xml_species_root = xml_species_tree.getroot()

            # Find the species element in the simulation XML tree
            xml_species_element = xml_species_root.find(f"SpeciesForest[@idSP='{species_id}']")
            if xml_species_element is None:
                raise ValueError(f"Species with ID '{species_id}' not found in simulation XML.")

            species_count = len(doc_species_forest_set.findall("SpeciesForest"))
            spec_id = species_count + 1
            new_species_element = copy.deepcopy(xml_species_element)
            new_species_element.set("idSP", str(spec_id))

            for event in new_species_element.findall("EventQ/Event"):
                event.set("idSP", str(spec_id))

            for tyf_params in new_species_element.findall("Growth/TYFParameters"):
                tyf_params.set("idSP", str(spec_id))

            doc_species_forest_set.append(new_species_element)
            doc_species_forest_set.set("count", str(spec_id))

            eventQ = doc_root.find("EventQ")
            events_count = len(eventQ.findall("Event"))
            eventQ.set("count", str(events_count + 1))
            xml_plant_event = xml_species_element.find(f"EventQ/Event[@nmEV='{plant_event_name}']")
            if xml_plant_event is not None:
                new_event = copy.deepcopy(xml_plant_event)
                new_event.set("regimeInstance", "1")
                new_event.set("nmRegime", "New Regime")
                new_event.set("idSP", str(spec_id))
                new_event.set("tEvent", "Doc")
                new_event.set("nYrsFromStEV", "0")
                new_event.set("nDaysFromStEV", "0")
                dateEV = Element("dateEV", {"CalendarSystemT": "FixedLength"})
                dateEV.text = plant_date.strftime("%Y%m%d")
                # new_event.append(dateEV)
                eventQ.insert(0, new_event)
                new_event.insert(1, dateEV)

            logger.info(
                f"Applied species XML content to simulation '{self.name}' for species ID '{species_id}'"
            )
        except Exception as e:
            logger.error(f"Failed to apply species XML: {str(e)}")
            raise FullCAMClientError(f"Failed to apply species XML: {str(e)}") from e

    def save_to_plo(self, file_path: str) -> None:
        """
        Save the simulation XML content to a .plo file.

        Args:
            file_path: Path to save the .plo file

        Raises:
            ValueError: If the simulation doesn't have XML content
        """
        if not self.doc_tree:
            raise ValueError(f"Simulation '{self.name}' doesn't have XML content.")

        try:
            root = self.doc_tree.getroot()

            # set the page to the about tab
            root.set("pageIxDO", "0")
            ET.indent(self.doc_tree)
            self.doc_tree.write(
                file_path,
                encoding="utf-8",
                xml_declaration=True,
                method="xml",
            )

            # Update plot_file attribute
            self.plot_file = file_path
            self.metadata["plot_file"] = file_path
            logger.info(f"Saved simulation '{self.name}' to plo file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save plo file: {str(e)}")
            raise FullCAMClientError(f"Failed to save plo file: {str(e)}") from e
