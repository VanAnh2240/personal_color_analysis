"""
classification/palettes.py
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple

RGB = Tuple[int, int, int]


@dataclass
class SeasonPalette:
    name: str
    hue: str         # 'warm' | 'cool'
    chroma: str      # 'bright' | 'muted' 
    value: str       # 'light' | 'dark'
    contrast: str    # 'high' | 'low'
    colors: List[RGB] = field(default_factory=list)

    @property
    def metric_vector(self) -> Tuple[int, int, int, int]:
        """
          S (Subtone / hue)    : warm=1, cool=0
          I (Intensity / chroma): bright=1, muted=0
          V (Value)            : light=1, dark=0
          C (Contrast)         : high=1,  low=0
        """
        return (
            1 if self.hue == "warm" else 0,      # S
            1 if self.chroma == "bright" else 0,  # I
            1 if self.value == "light" else 0,    # V
            1 if self.contrast == "high" else 0,  # C
        )


# SPRING  : 1111 
SPRING = SeasonPalette(
    name="Spring",
    hue="warm",
    chroma="bright",
    value="light",
    contrast="high",
    colors=[
        ( 28,  46, 112),   
        ( 60, 137, 188),  
        (126, 174,  53),   
        (115, 189, 168),  
        (101,  61, 126),  
        (246,  68,  44), 
        (245,  89,  72), 
        (251, 134,  48),  
        (253, 230,  55),  
        (230, 174,  91), 
    ],
)


# SUMMER  SIVC: 0010 
SUMMER = SeasonPalette(
    name="Summer",
    hue="cool",
    chroma="muted",
    value="light",
    contrast="low",
    colors=[
        (163, 206, 222),   
        (153, 141, 179),  
        ( 77,  77, 120), 
        ( 55,  51,  87), 
        ( 29, 163, 148),  
        (151, 142, 137),
        (124, 116, 169),
        (185,  68, 137), 
        (202,  61,  79), 
        (241, 175, 193), 
    ],
)

# AUTUMN  SIVC: 1000
AUTUMN = SeasonPalette(
    name="Autumn",
    hue="warm",
    chroma="muted",
    value="dark",
    contrast="low",
    colors=[
        ( 59,  68,  52),  
        (142, 115,  61), 
        ( 69,  69,  70),  
        ( 63,  46,  50),  
        (111,  55,  48), 
        (248,  90,  89), 
        (208,  58,  69),
        (220, 101,  78),  
        (249, 194,  91),  
        (168, 114,  65), 
    ],
)


# WINTER  SIVC: 0101 
WINTER = SeasonPalette(
    name="Winter",
    hue="cool",
    chroma="bright",
    value="dark",
    contrast="high",
    colors=[
        (255, 246, 107),  
        ( 26,  21,  22),  
        ( 47,  27,  76),  
        ( 55, 118, 179),  
        (254, 249, 237), 
        ( 18,  15,   6),  
        (214,  50,  49),   
        (158,  23,  76),  
        (220,  49,  95),  
        ( 39, 100,  14), 
    ],
)

ALL_SEASONS: List[SeasonPalette] = [SPRING, SUMMER, AUTUMN, WINTER]

SEASON_MAP = {s.name.lower(): s for s in ALL_SEASONS}