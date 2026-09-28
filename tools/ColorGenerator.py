import matplotlib.colors as mcolors
import random

default_base_colors = list(mcolors.TABLEAU_COLORS.values())

def rgb_to_cmyk(rgb):
    r, g, b = [x / 255.0 for x in rgb]
    k = 1 - max(r, g, b)
    if k == 1:
        return (0, 0, 0, 1)
    c = (1 - r - k) / (1 - k)
    m = (1 - g - k) / (1 - k)
    y = (1 - b - k) / (1 - k)
    return (c, m, y, k)

def to_rgb(hex_color):
    return tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))

def format_color(color, scale):
    if scale == 'matplotlib':
        return color
    elif scale == 'rgb':
        return to_rgb(color)
    elif scale == 'cmyk':
        return rgb_to_cmyk(to_rgb(color))
    elif scale == 'hex':
        return color
    else:
        raise ValueError("Unsupported scale. Choose from 'matplotlib', 'rgb', 'cmyk', or 'hex'.")

class ColorGenerator:
    def __init__(self, scale='matplotlib', colors='default', combination_hierarchy='same as stypes'):
        self.scale = scale
        self.color_map = {} if colors == 'default' else colors
        self.combination_hierarchy = list(self.color_map.keys()) if combination_hierarchy == 'same as stypes' else combination_hierarchy
        self.available_colors = iter(default_base_colors)

    def assign_color(self, stype):
        if stype not in self.color_map:
            try:
                color = next(self.available_colors)
            except StopIteration:
                color = "#" + ''.join(random.choices('0123456789ABCDEF', k=6))  # Generate random hex color
            self.color_map[stype] = color
            self.combination_hierarchy.append(stype)  # Add to hierarchy in order of assignment
        return self.color_map[stype]

    def resolve_combination(self, stype):
        if self.combination_hierarchy:
            for s in self.combination_hierarchy:
                if s in stype:
                    return self.color_map.get(s, self.assign_color(s))
        return self.assign_color(stype)

    def get_color(self, stype):
        if isinstance(stype, str):
            color = self.assign_color(stype)
        elif isinstance(stype, (tuple, list)):
            color = self.resolve_combination(stype)
        else:
            raise ValueError("stype must be a string or a tuple/list of strings.")
        return format_color(color, self.scale)