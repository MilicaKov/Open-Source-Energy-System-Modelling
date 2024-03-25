import math
from main import calculate_circle_properties, calculate_rectangle_properties, calculate_triangle_properties
""" Testing the functions """

def test_calculate_circle_properties():
    area, circumference = calculate_circle_properties(5)
    assert math.isclose(area, math.pi * 25, rel_tol=1e-9)
    assert math.isclose(circumference, 2 * math.pi * 5, rel_tol=1e-9)

def test_calculate_rectangle_properties():
    area, perimeter = calculate_rectangle_properties(4, 6)
    assert area == 24
    assert perimeter == 20

def test_calculate_triangle_properties():
    area, perimeter = calculate_triangle_properties(3, 4, 5)
    assert math.isclose(area, 6, rel_tol=1e-9)  # Using Heron's formula
    assert perimeter == 12
