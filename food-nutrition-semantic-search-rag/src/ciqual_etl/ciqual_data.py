#!/usr/bin/env python3
"""Dataclasses representing Ciqual XML data structures."""

from typing import Optional
from dataclasses import dataclass

# Use @dataclass decorator to automatically generate boilerplate code for class data (e.g., __init__).
@dataclass
class FoodGroup:
    """Food group from `alim_grp_2025_11_03.xml` XML file."""
    alim_grp_code: str
    alim_ssgrp_code: str
    alim_ssssgrp_code: str
    alim_grp_nom_eng: Optional[str]
    alim_ssgrp_nom_eng: Optional[str]
    alim_ssssgrp_nom_eng: Optional[str]
    alim_grp_nom_fr: Optional[str]
    alim_ssgrp_nom_fr: Optional[str]
    alim_ssssgrp_nom_fr: Optional[str]

@dataclass
class Food:
    """Food item from `alim_2025_11_03.xml` XML file."""
    alim_code: int
    alim_nom_eng: Optional[str]
    alim_nom_fr: Optional[str]
    alim_nom_sci: Optional[str]
    facteur_jones: Optional[float]
    alim_grp_code: str
    alim_ssgrp_code: str
    alim_ssssgrp_code: str

@dataclass
class Component:
    """Nutrition component from `const_2025_11_03.xml` XML file."""
    const_code: int
    const_nom_eng: Optional[str]
    const_nom_fr: Optional[str]
    code_infoods: Optional[str]

@dataclass
class Composition:
    """Nutrition composition data from `compo_2025_11_03.xml` XML file."""
    alim_code: int
    const_code: int
    teneur: Optional[str]
    min_val: Optional[str]
    max_val: Optional[str]
    code_confiance: Optional[str]
    source_code: Optional[int]

@dataclass
class DataSource:
    """Data source from `sources_2025_11_03.xml` XML file."""
    source_code: int
    ref_citation: str