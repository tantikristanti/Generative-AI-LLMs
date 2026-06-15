#!/usr/bin/env python3
"""XML parser for Ciqual 2025 files."""

import os
import logging
import xml.etree.ElementTree as ET
from typing import List, Optional
from ciqual_etl import FoodGroup, Food, Component, Composition, DataSource

logger = logging.getLogger(__name__)

class CiqualXMLParser:
    """
    Parser for the Ciqual 2025 food composition XML files.

    This class reads the XML files provided by ANSES (Ciqual table) and converts them into Python dataclass objects (Food, FoodGroup, Component, Composition, DataSource). 
    The parser handles missing files, missing elements, and type conversions (int, float) gracefully.
    """

    def __init__(self, xml_dir: str):
        """
        Initialize the parser with the directory containing the XML files.

        Args:
            xml_dir (str): Path to the directory where the Ciqual 2025 XML files are stored (e.g., './data/ciqual/').
        """
        
        self.xml_dir = xml_dir
        self.namespaces = {
            'ciqual': ''  # The XML files might not use namespaces
        }
        
    def parse_xml_file(self, file_path: str, row_tag: str) -> List[ET.Element]:
        """
        Parse an XML file and return a list of row elements matching the tag.

        Args:
            file_path (str): Name of the XML file (e.g., 'alim_2025_11_03.xml').
            row_tag (str): XML tag of the repeating row elements (e.g., 'ALIM').

        Returns:
            List[ET.Element]: List of XML elements found under the root with the given tag. 
            Returns an empty list if the file does not exist.

        Note:
            The method uses XPath `.//{row_tag}` to find all elements regardless of their depth.
        """
        
        full_path = os.path.join(self.xml_dir, file_path)
        if not os.path.exists(full_path):
            logger.warning(f"File not found: {full_path}")
            return []

        tree = ET.parse(full_path)
        root = tree.getroot()
        return root.findall(f".//{row_tag}")

    def parse_food_groups(self) -> List[FoodGroup]:
        """
        Parse the food group hierarchy from alim_grp_2025_11_03.xml.

        The file contains three levels: group, sub-group, and sub-sub-group,
        each with French and English names.

        Returns:
            List[FoodGroup]: A list of FoodGroup dataclass instances,
            one per XML entry. Empty list if the file is missing or has no data.

        Logs:
            INFO: Number of food groups parsed.
        """
        
        elements = self.parse_xml_file('alim_grp_2025_11_03.xml', 'ALIM_GRP')
        food_groups = []

        for elem in elements:
            food_groups.append(FoodGroup(
                alim_grp_code=self._get_text(elem, 'alim_grp_code'),
                alim_ssgrp_code=self._get_text(elem, 'alim_ssgrp_code'),
                alim_ssssgrp_code=self._get_text(elem, 'alim_ssssgrp_code'),
                alim_grp_nom_eng=self._get_text(elem, 'alim_grp_nom_eng'),
                alim_ssgrp_nom_eng=self._get_text(elem, 'alim_ssgrp_nom_eng'),
                alim_ssssgrp_nom_eng=self._get_text(elem, 'alim_ssssgrp_nom_eng'),
                alim_grp_nom_fr=self._get_text(elem, 'alim_grp_nom_fr'),
                alim_ssgrp_nom_fr=self._get_text(elem, 'alim_ssgrp_nom_fr'),
                alim_ssssgrp_nom_fr=self._get_text(elem, 'alim_ssssgrp_nom_fr')
            ))

        logger.info(f"Parsed {len(food_groups)} food groups")
        return food_groups

    def parse_foods(self) -> List[Food]:
        """
        Parse the food items from alim_2025_11_03.xml.

        Each food entry includes a unique code, English/French/Scientific names,
        a Jones factor, and references to the group hierarchy.

        Returns:
            List[Food]: A list of Food dataclass instances. Foods without an
            `alim_code` are silently skipped. Empty list if the file is missing.

        Logs:
            INFO: Number of foods parsed.
        """
        
        elements = self.parse_xml_file('alim_2025_11_03.xml', 'ALIM')
        foods = []

        for elem in elements:
            alim_code = self._get_text(elem, 'alim_code')
            if not alim_code:
                continue

            foods.append(Food(
                alim_code=int(alim_code),
                alim_nom_eng=self._get_text(elem, 'alim_nom_eng'),
                alim_nom_fr=self._get_text(elem, 'alim_nom_fr'),
                alim_nom_sci=self._get_text(elem, 'alim_nom_sci'),
                facteur_jones=self._get_float(elem, 'facteur_jones'),
                alim_grp_code=self._get_text(elem, 'alim_grp_code'),
                alim_ssgrp_code=self._get_text(elem, 'alim_ssgrp_code'),
                alim_ssssgrp_code=self._get_text(elem, 'alim_ssssgrp_code')
            ))

        logger.info(f"Parsed {len(foods)} foods")
        return foods

    def parse_components(self) -> List[Component]:
        """
        Parse the nutrient/component definitions from const_2025_11_03.xml.

        Components are the nutritional measures (e.g., energy, protein, fat)
        that can be reported for each food.

        Returns:
            List[Component]: A list of Component dataclass instances.
            Components without a `const_code` are ignored. Empty list if the
            file is missing.

        Logs:
            INFO: Number of components parsed.
        """
        
        elements = self.parse_xml_file('const_2025_11_03.xml', 'CONST')
        components = []

        for elem in elements:
            const_code = self._get_text(elem, 'const_code')
            if not const_code:
                continue

            components.append(Component(
                const_code=int(const_code),
                const_nom_eng=self._get_text(elem, 'const_nom_eng'),
                const_nom_fr=self._get_text(elem, 'const_nom_fr'),
                code_infoods=self._get_text(elem, 'code_INFOODS')
            ))

        logger.info(f"Parsed {len(components)} components")
        return components

    def parse_composition(self) -> List[Composition]:
        """
        Parse the nutrient composition data from compo_2025_11_03.xml.

        This file links foods (by alim_code) to components (by const_code) and
        gives the measured value (teneur), possible min/max, confidence code,
        and a reference to the data source.

        Returns:
            List[Composition]: A list of Composition dataclass instances.
            Entries missing either `alim_code` or `const_code` are skipped.
            If `source_code` is present, it is stored as an int, else None.

        Logs:
            INFO: Number of composition records parsed.
        """
        
        elements = self.parse_xml_file('compo_2025_11_03.xml', 'COMPO')
        compositions = []

        for elem in elements:
            alim_code = self._get_text(elem, 'alim_code')
            const_code = self._get_text(elem, 'const_code')
            if not alim_code or not const_code:
                continue

            source_code = self._get_text(elem, 'source_code')

            compositions.append(Composition(
                alim_code=int(alim_code),
                const_code=int(const_code),
                teneur=self._get_text(elem, 'teneur'),
                min_val=self._get_text(elem, 'min'),
                max_val=self._get_text(elem, 'max'),
                code_confiance=self._get_text(elem, 'code_confiance'),
                source_code=int(source_code) if source_code else None
            ))

        logger.info(f"Parsed {len(compositions)} composition records")
        return compositions

    def parse_data_sources(self) -> List[DataSource]:
        """
        Parse the data source references from sources_2025_11_03.xml.

        Each source_code corresponds to a literature reference or database
        used to obtain the composition values.

        Returns:
            List[DataSource]: A list of DataSource dataclass instances.
            Sources without a `source_code` are ignored. Empty list if the
            file is missing.

        Logs:
            INFO: Number of data sources parsed.
        """
        
        elements = self.parse_xml_file('sources_2025_11_03.xml', 'SOURCES')
        sources = []

        for elem in elements:
            source_code = self._get_text(elem, 'source_code')
            if not source_code:
                continue

            sources.append(DataSource(
                source_code=int(source_code),
                ref_citation=self._get_text(elem, 'ref_citation')
            ))

        logger.info(f"Parsed {len(sources)} data sources")
        return sources

    @staticmethod
    def _get_text(element: ET.Element, tag: str) -> Optional[str]:
        """
        Safely extract the text content of a child XML element.

        Args:
            element (ET.Element): Parent XML element.
            tag (str): Child tag name to look for.

        Returns:
            Optional[str]: The stripped text of the child element if it exists
            and has non‑empty text; otherwise None.
        """
        
        child = element.find(tag)
        return child.text.strip() if child is not None and child.text else None

    @staticmethod
    def _get_float(element: ET.Element, tag: str) -> Optional[float]:
        """
        Safely extract a floating‑point number from a child XML element.

        Args:
            element (ET.Element): Parent XML element.
            tag (str): Child tag name to look for.

        Returns:
            Optional[float]: The parsed float value if the child exists and
            its text can be converted; otherwise None. ValueError during
            conversion is caught and returns None.

        Note:
            This method uses `_get_text` internally; empty or missing text
            will lead to None.
        """
        
        text = CiqualXMLParser._get_text(element, tag)
        if text:
            try:
                return float(text)
            except ValueError:
                return None
        return None