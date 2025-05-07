"""
Simulation Manager for FullCAM client

This module provides classes for managing FullCAM plot simulations,
comparing their results, and creating visualizations.
"""

import io
import logging
import os
from datetime import datetime

import pandas as pd
import pyarrow as pa
from pyarrow import csv
from dataclasses import dataclass
import xml.etree.ElementTree as ET
from typing import Optional, Union
from xml.etree.ElementTree import Element
from fullcam_client.exceptions import FullCAMClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("fullcam_client.simulation")

@dataclass
class Build:
    def __init__(self, build: Element):
        latitude= build.get("latBL")
        longitude= build.get("lonBL")   

    latitude: float = 0.0
    longitude: float = 0.0 

@dataclass
class Timing:
    def __init__(self, timing: Element):
        start_date_el= timing.find("stDateTM")
        end_date_el= timing.get("enDateTM")
        if start_date_el is not None:
            start_date= datetime.strptime(start_date_el.text, "%Y%m%d").date()
        if end_date_el is not None:
            end_date= datetime.strptime(end_date_el.text, "%Y%m%d").date()

    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

@dataclass
class About:
    def __init__(self, meta: Element):
        name= meta.get("nmME")
        notes_el= meta.find("notesME")
        if notes_el is not None:
            notes= notes_el.text
        else:
            notes= None
        version= meta.get("savedByVersion")

    name: str = ""
    notes: Optional[str] = None 
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
                if start_date_el is not None and self.timing.start_date:
                    start_date_el.text = self.timing.start_date.strftime("%Y%m%d")
                else:
                    logger.warning("Start date element not found in XML. Skipping start date update.")

                end_date_el = timing.find("enDateTM")
                if end_date_el is not None and self.timing.end_date:
                    end_date_el.text = self.timing.end_date.strftime("%Y%m%d")
                else:
                    logger.warning("End date element not found in XML. Skipping end date update.")

            build = root.find("Build")
            if build is not None:
                build.set("latBL", str(self.build.latitude))
                build.set("lonBL", str(self.build.longitude))
            else:
                logger.warning("Build element not found in XML. Skipping latitude and longitude update.")


            ET.indent(self.tree)
            self.tree.write(file_path, encoding="utf-8", xml_declaration=True, method='xml')
            #with open(file_path, "w", encoding="utf-8") as f:
            #    f.write(self.xml_content)

            # Update plot_file attribute
            self.plot_file = file_path
            self.metadata["plot_file"] = file_path
            logger.info(f"Saved simulation '{self.name}' to plo file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save plo file: {str(e)}")
            raise FullCAMClientError(f"Failed to save plo file: {str(e)}") from e
