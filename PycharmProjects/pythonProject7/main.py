import math

def calculate_circle_properties(radius):
    """
    Calculates properties of a circle given its radius. Arguments: radius (float).
    Returns: A tuple containing the area and circumference of the circle.
    """
    area = math.pi * radius ** 2
    circumference = 2 * math.pi * radius
    return area, circumference

def calculate_rectangle_properties(length, width):
    """
    Calculates properties of a rectangle given its length and width.
    Args: length (float), width (float).
    Returns: A tuple containing the area and perimeter of the rectangle.
    """
    area = length * width
    perimeter = 2 * (length + width)
    return area, perimeter

def calculate_triangle_properties(side1, side2, side3):
    """
    Calculates properties of a triangle given its three sides.
    Args:
        side1 (float): Length of the first side.
        side2 (float): Length of the second side.
        side3 (float): Length of the third side.
    Returns: A tuple containing the area and perimeter of the triangle.
    """
    semi_perimeter = (side1 + side2 + side3) / 2
    area = math.sqrt(semi_perimeter * (semi_perimeter - side1) * (semi_perimeter - side2) * (semi_perimeter - side3))
    perimeter = side1 + side2 + side3
    return area, perimeter

def calculate_square_properties(side):
    """
        Calculates properties of a square given its length.
        Args: length (float).
        Returns: A tuple containing the area and perimeter of the square.
        """
    AREA=side*side
    PERIMETER=(4*(side))
    return AREA,PERIMETER

circle_area, circle_circumference = calculate_circle_properties(5)
print("Circle - Area:", circle_area, "Circumference:", circle_circumference)

rectangle_area, rectangle_perimeter = calculate_rectangle_properties(4, 6)
print("Rectangle - Area:", rectangle_area, "Perimeter:", rectangle_perimeter)

triangle_area, triangle_perimeter = calculate_triangle_properties(3, 4, 5)
print("Triangle - Area:", triangle_area, "Perimeter:", triangle_perimeter)

square_area, square_perimeter = calculate_square_properties(3)
print("Square - Area:", square_area, "Perimeter:", square_perimeter)



