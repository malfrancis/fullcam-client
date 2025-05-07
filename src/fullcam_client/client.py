"""
Client for interacting with the FullCAM Plot API
"""

import io
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import pyarrow as pa
import requests
from pyarrow import csv

from fullcam_client.exceptions import FullCAMAPIError
from fullcam_client.simulation import Simulation

# Configure logging
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("fullcam_client")


@dataclass
class Template:
    """
    Represents a FullCAM template from the API

    Attributes:
        id: Unique identifier for the template
        value: File path/name of the template
        name: Template name
        notes: Additional notes about the template
        saved_by_research: Whether the template was saved by research
        saved_by_version: Version information for the template
        lock_time: Time when the template was locked
        lock_id: ID of the lock
        lock_on_me: Whether the template is locked on ME
    """

    id: str
    value: str
    name: str
    notes: str = ""


class FullCAMClient:
    """
    Client for interacting with the FullCAM APIs

    This client handles:
    1. Loading XML plot files
    2. Sending XML to the FullCAM Plot API
    3. Processing responses into Arrow format
    4. Converting results between formats (CSV, dataframes)

    Attributes:
        api_url: URL for the FullCAM Plot API
        subscription_key: Subscription key for the API"""

    DEFAULT_API_URL_2020 = "https://api.climatechange.gov.au/climate/carbon-accounting/2020"
    DEFAULT_API_URL_2024 = "https://api.climatechange.gov.au/climate/carbon-accounting/2024"
    SUBSCRIPTION_KEY_2020 = "8505062cc2a94306878c33e197d1fa67"  # Replace with your actual
    SUBSCRIPTION_KEY_2024 = "5086b09defed4a16850cb87fd63fef7c"  # Replace with your actual
    # subscription key

    def __init__(self, version: str = "2020"):
        """
        Initialize the FullCAM client

        Args:
            api_url: URL for the FullCAM Plot API. If not provided, will use the default API URL.
        """
        if version not in ["2020", "2024"]:
            raise ValueError("Version must be either '2020' or '2024'")
        if version == "2024":
            self.api_url = self.DEFAULT_API_URL_2024
            self.subscription_key = self.SUBSCRIPTION_KEY_2024
            logger.info(f"Initialized FullCAM client with API URL: {self.api_url}")
        else:
            # Default to 2020 API URL and subscription key
            logger.info("Using default API URL for 2020")
            self.api_url = self.DEFAULT_API_URL_2020
            self.subscription_key = self.SUBSCRIPTION_KEY_2020
            logger.info(f"Initialized FullCAM client with API URL: {self.api_url}")

    def simulate(self, plot_xml: str) -> pa.Table:
        """
        Send XML to the FullCAM Plot API to simulate and get results

        Returns:
            CSV data as a string

        Raises:
            FullCAMClientError: If no XML content has been loaded
            FullCAMAPIError: If the API request fails
        """

        # Prepare the form data
        # Ensure plot_xml is properly encoded
        if isinstance(plot_xml, str):
            plot_xml_bytes = plot_xml.encode("utf-8")
        else:
            plot_xml_bytes = plot_xml  # Assume it's already bytes
        files = {"file": ("plot.plo", plot_xml_bytes, "application/xml")}

        # Define headers
        headers = {"Ocp-Apim-Subscription-Key": self.subscription_key}

        plot_sim_url = self.api_url + "/plot/v1/2020/fullcam-simulator/run-plotsimulation"

        # Make the API request
        logger.info("Sending request to FullCAM Plot API")
        try:
            response = requests.post(plot_sim_url, headers=headers, files=files)

            # Check for successful response
            if response.status_code != 200:
                logger.error(f"API request failed with status code {response.status_code}")
                logger.error(f"Response content: {response.text}")
                raise FullCAMAPIError(
                    "Failed to get response from FullCAM API",
                    status_code=response.status_code,
                    response_text=response.text,
                )

            # Convert to Arrow table immediately and store it
            csv_stream = io.BytesIO(response.text.encode())
            output_table = csv.read_csv(csv_stream)
            logger.info(f"Converted API response to Arrow table with {len(output_table)} rows")

            # Return the CSV data from the response without storing it
            return output_table

        except requests.RequestException as e:
            logger.error(f"Request exception: {str(e)}")
            raise FullCAMAPIError(f"Failed to connect to FullCAM API: {str(e)}") from e

    def get_templates(self) -> list[Template]:
        """
        Get the list of available templates from the FullCAM Plot API

        Returns:
            List of Template objects containing template information

        Raises:
            FullCAMAPIError: If the API request fails
        """
        # Define headers
        headers = {"Ocp-Apim-Subscription-Key": self.subscription_key}
        templates_url = f"{self.api_url}/data/v1/2020/data-builder/templates?version=2020"
        logger.info("Fetching available templates from the FullCAM API")
        try:
            response = requests.get(templates_url, headers=headers)

            # Check for successful response
            if response.status_code != 200:
                logger.error(f"API request failed with status code {response.status_code}")
                raise FullCAMAPIError(
                    "Failed to get templates from FullCAM API",
                    status_code=response.status_code,
                    response_text=response.text,
                )

            xml_stream = io.BytesIO(response.text.encode())
            tree = ET.parse(xml_stream)
            root = tree.getroot()

            templates = []

            # Find the ItemList element with id="Templates"
            templates_list = root.find(".//ItemList[@id='Templates']")
            if templates_list is not None:
                # Process each ItemInfo element
                for item in templates_list.findall("./ItemInfo"):
                    template_id = item.get("id")
                    template_value = item.get("value")

                    # Get Meta element data
                    meta_element = item.find("./Meta")
                    if meta_element is not None:
                        template_name = meta_element.get("nmME", "")

                        # Get notes if available
                        notes_element = meta_element.find("./notesME")
                        notes = (
                            notes_element.text
                            if notes_element is not None and notes_element.text
                            else ""
                        )

                        # Create Template object
                        template = Template(
                            id=template_id, value=template_value, name=template_name, notes=notes
                        )

                        templates.append(template)

            logger.info(f"Found {len(templates)} templates")
            return templates

        except requests.RequestException as e:
            logger.error(f"Request exception: {str(e)}")
            raise FullCAMAPIError(f"Failed to connect to FullCAM API: {str(e)}") from e
        except ET.ParseError as e:
            logger.error(f"XML parse error: {str(e)}")
            raise FullCAMAPIError(f"Failed to parse XML response: {str(e)}") from e

    def get_template_xml(self, template: Template | str) -> Simulation:
        """
        Create a simulation from a template

        Args:
            template: Template object to use for the simulation

        Returns:
            Simulation object initialized with the template
        """
        if isinstance(template, Template):
            template_name = template.value
        elif isinstance(template, str):
            template_name = template
        else:
            raise ValueError("Invalid template provided")

        headers = {"Ocp-Apim-Subscription-Key": self.subscription_key}
        templates_url = (
            f"{self.api_url}/data/v1/2020/data-builder/template?templateName={template_name}"
        )
        logger.info("Fetching template from the FullCAM API")
        try:
            response = requests.get(templates_url, headers=headers)

            # Check for successful response
            if response.status_code != 200:
                logger.error(f"API request failed with status code {response.status_code}")
                raise FullCAMAPIError(
                    "Failed to get template from FullCAM API",
                    status_code=response.status_code,
                    response_text=response.text,
                )

            xml_stream = io.BytesIO(response.text.encode())
            tree = ET.parse(xml_stream)
            root = tree.getroot()
            doc = root.find("DocumentPlot")
            log_entry_set = doc.find("LogEntrySet")
            if log_entry_set is not None:
                log_entry_set.set("count", "0")
                for entry in log_entry_set.findall("LogEntry"):
                    # Remove the LogEntry element from the parent
                    log_entry_set.remove(entry)

            txt = ET.tostring(
                doc, encoding="utf-8", method="xml", xml_declaration=True, short_empty_elements=True
            ).decode("utf-8")
            return txt

        except requests.RequestException as e:
            logger.error(f"Request exception: {str(e)}")
            raise FullCAMAPIError(f"Failed to connect to FullCAM API: {str(e)}") from e
        except ET.ParseError as e:
            logger.error(f"XML parse error: {str(e)}")
            raise FullCAMAPIError(f"Failed to parse XML response: {str(e)}") from e

    def get_location_xml(
        self,
        latitude: float,
        longitude: float,
        area: str = "Cell",
        plot_type: str = "CompF",
        forest_category: str = "ERF",
        include_growth: bool = True,
        version: int = 2020,
    ) -> str:
        """
            area=Cell&plotT=CompF&frCat=ERF&incGrowth=true' \
        """

        headers = {"Ocp-Apim-Subscription-Key": self.subscription_key}
        templates_url = f"{self.api_url}/data/v1/2020/data-builder/siteinfo?latitude={latitude}&longitude={longitude}&area={area}&plotT={plot_type}&frCat={forest_category}&incGrowth={include_growth}"
        logger.info("Fetching site info from the FullCAM API")
        try:
            response = requests.get(templates_url, headers=headers)

            # Check for successful response
            if response.status_code != 200:
                logger.error(f"API request failed with status code {response.status_code}")
                raise FullCAMAPIError(
                    "Failed to get template from FullCAM API",
                    status_code=response.status_code,
                    response_text=response.text,
                )

            xml_stream = io.BytesIO(response.text.encode())
            tree = ET.parse(xml_stream)
            doc_fragment = tree.getroot()
            if doc_fragment is not None:
                if version == 2020 and doc_fragment.get("Version") != "5007":
                    logger.error(f"API request failed with status code {response.status_code}")
                    raise FullCAMAPIError(
                        f"Ivalid document fragment version {doc_fragment.get('Version')} for 2020",
                    )

            txt = ET.tostring(
                doc_fragment,
                encoding="utf-8",
                method="xml",
                xml_declaration=True,
                short_empty_elements=True,
            ).decode("utf-8")
            return txt

        except requests.RequestException as e:
            logger.error(f"Request exception: {str(e)}")
            raise FullCAMAPIError(f"Failed to connect to FullCAM API: {str(e)}") from e
        except ET.ParseError as e:
            logger.error(f"XML parse error: {str(e)}")
            raise FullCAMAPIError(f"Failed to parse XML response: {str(e)}") from e

    def create_simulation_from_template(
        self, template: Template | str, simulation_name: str = None
    ) -> "Simulation":
        # Get template XML
        xml_content = self.get_template_xml(template)

        # If name not provided, use template id/name
        if simulation_name is None:
            simulation_name = (
                "Simulation_" + template.value
                if isinstance(template, Template)
                else "Simulation_" + template
            )

        # Import here to avoid circular import
        from fullcam_client.simulation import Simulation

        # Create and return a Simulation object
        return Simulation(name=simulation_name, xml_content=xml_content, client=self)
