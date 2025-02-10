# annotator/utils.py
def point_in_polygon(x, y, poly_points):
    """
    Determine if point (x, y) is inside the polygon defined by poly_points.
    poly_points: list of (x, y) tuples.
    Returns True if inside, False otherwise.
    """
    inside = False
    n = len(poly_points)
    j = n - 1
    for i in range(n):
        xi, yi = poly_points[i]
        xj, yj = poly_points[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi):
            inside = not inside
        j = i
    return inside
