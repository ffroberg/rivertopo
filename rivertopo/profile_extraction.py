
from osgeo import gdal, ogr
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from collections import namedtuple
import argparse
import os

from rivertopo.cross_lines_z import *

gdal.UseExceptions()
ogr.UseExceptions()

"""
This script is used to extract elevation profiles along perpendicular lines to a given polyline 
feature in a DEM raster

"""

# Create bounding box

### copied sampling.py from hydroadjust ###

BoundingBox = namedtuple(
    'BoundingBox',
    ['x_min', 'x_max', 'y_min', 'y_max'],
)

def get_raster_window(dataset, bbox):
    """
    Return a window of the input raster dataset, containing at least the
    provided bounding box.
    
    :param dataset: Source raster dataset
    :type dataset: GDAL Dataset object
    :param bbox: Window bound coordinates
    :type bbox: hydroadjust.sampling.BoundingBox object
    :returns: GDAL Dataset object for the requested window
    """
    
    input_geotransform = dataset.GetGeoTransform()
    
    if input_geotransform[2] != 0.0 or input_geotransform[4] != 0.0:
        raise ValueError("geotransforms with rotation are unsupported")
    
    input_offset_x = input_geotransform[0]
    input_offset_y = input_geotransform[3]
    input_pixelsize_x = input_geotransform[1]
    input_pixelsize_y = input_geotransform[5]
    
    # We want to find window coordinates that:
    # a) are aligned to the source raster pixels
    # b) contain the requested bounding box plus at least one pixel of "padding" on each side, to allow for small floating-point rounding errors in X/Y coordinates
    # 
    # Recall that the pixel size in the geotransform is commonly negative, hence all the min/max calls.
    raw_x_min_col_float = (bbox.x_min - input_offset_x) / input_pixelsize_x
    raw_x_max_col_float = (bbox.x_max - input_offset_x) / input_pixelsize_x
    raw_y_min_row_float = (bbox.y_min - input_offset_y) / input_pixelsize_y
    raw_y_max_row_float = (bbox.y_max - input_offset_y) / input_pixelsize_y
    
    col_min = int(np.floor(min(raw_x_min_col_float, raw_x_max_col_float))) - 1
    col_max = int(np.ceil(max(raw_x_min_col_float, raw_x_max_col_float))) + 1
    row_min = int(np.floor(min(raw_y_min_row_float, raw_y_max_row_float))) - 1
    row_max = int(np.ceil(max(raw_y_min_row_float, raw_y_max_row_float))) + 1
    
    x_col_min = input_offset_x + input_pixelsize_x * col_min
    x_col_max = input_offset_x + input_pixelsize_x * col_max
    y_row_min = input_offset_y + input_pixelsize_y * row_min
    y_row_max = input_offset_y + input_pixelsize_y * row_max
    
    # Padded, georeferenced window coordinates. The target window to use with gdal.Translate().
    padded_bbox = BoundingBox(
        x_min=min(x_col_min, x_col_max),
        x_max=max(x_col_min, x_col_max),
        y_min=min(y_row_min, y_row_max),
        y_max=max(y_row_min, y_row_max),
    )
    
    # Size in pixels of destination raster
    dest_num_cols = col_max - col_min
    dest_num_rows = row_max - row_min
    
    translate_options = gdal.TranslateOptions(
        width=dest_num_cols,
        height=dest_num_rows,
        projWin=(padded_bbox.x_min, padded_bbox.y_max, padded_bbox.x_max, padded_bbox.y_min),
        resampleAlg=gdal.GRA_NearestNeighbour,
    )
    
    # gdal.Translate() needs a destination *name*, not just a Dataset to
    # write into. Create a temporary file in GDAL's virtual filesystem as a
    # stepping stone.
    window_dataset_name = "/vsimem/temp_window.tif"
    window_dataset = gdal.Translate(
        window_dataset_name,
        dataset,
        options=translate_options
    )
    
    return window_dataset


def get_raster_interpolator(dataset):
    """
    Return a scipy.interpolate.RegularGridInterpolator corresponding to a GDAL
    raster.
    
    :param dataset: Raster dataset in which to interpolate
    :type dataset: GDAL Dataset object
    :returns: RegularGridInterpolator accepting georeferenced X and Y input
    """
    
    geotransform = dataset.GetGeoTransform()
    band = dataset.GetRasterBand(1)
    nodata_value = band.GetNoDataValue()
    z_grid = band.ReadAsArray()
    num_rows, num_cols = z_grid.shape
    
    if geotransform[2] != 0.0 or geotransform[4] != 0.0:
        raise ValueError("geotransforms with rotation are unsupported")
    
    # X and Y values for the individual columns/rows of the raster. The 0.5 is
    # added in order to obtain the coordinates of the cell centers rather than
    # the corners.
    x_values = geotransform[0] + geotransform[1]*(0.5+np.arange(num_cols))
    y_values = geotransform[3] + geotransform[5]*(0.5+np.arange(num_rows))

    # RegularGridInterpolator requires the x and y arrays to be in strictly
    # increasing order, accommodate this
    if geotransform[1] > 0.0:
        col_step = 1
    else:
        col_step = -1
        x_values = np.flip(x_values)

    if geotransform[5] > 0.0:
        row_step = 1
    else:
        row_step = -1
        y_values = np.flip(y_values)
        
    # NODATA values must be replaced with NaN for interpolation purposes
    z_grid[z_grid == nodata_value] = np.nan
    
    # The grid must be transposed to swap (row, col) coordinates into (x, y)
    # order.
    interpolator = RegularGridInterpolator(
        points=(x_values, y_values),
        values=z_grid[::row_step, ::col_step].transpose(),
        method='linear',
        bounds_error=False,
        fill_value=np.nan,
    )
    
    return interpolator

def calculate_directions_for_chainage_points(stream_linestring, chainage_points):
    """
    Calculate direction vectors for a list of chainage points along the stream linestring.
    
    :param stream_linestring: The input stream linestring (OGR geometry).
    :param chainage_points: A list of tuples, each containing the X and Y coordinates of a chainage point.
    :return: A list of tuples, each containing a chainage point and its direction vector.
    """
    directions = []
    tolerance = 0
    # Calculate the expanded bounding box of the stream linestring
    minx, miny, maxx, maxy = stream_linestring.GetEnvelope()
    bbox_expanded = (minx - tolerance, miny - tolerance, maxx + tolerance, maxy + tolerance)

    for point_geometry, chainage in chainage_points:
        x, y = point_geometry.GetX(), point_geometry.GetY()
        
        # Check if the chainage point is within the expanded bounding box
        if not (bbox_expanded[0] <= x <= bbox_expanded[2] and bbox_expanded[1] <= y <= bbox_expanded[3]):
            continue  # Skip this point as it's outside the tolerance area

        closest_distance = float('inf')
        direction_vector = (0, 0)
        
        for i in range(stream_linestring.GetPointCount() - 1):

            point1 = stream_linestring.GetPoint(i)[:2] 
            point2 = stream_linestring.GetPoint(i + 1)[:2]

            dx = point2[0] - point1[0]
            dy = point2[1] - point1[1]
            temp_direction_vector = (dx, dy)
            norm = np.linalg.norm([dx, dy])

            if norm != 0:  # Normalize
                normalized_temp_direction = (dx / norm, dy / norm)
            else:
                normalized_temp_direction = temp_direction_vector
            
            segment = ogr.Geometry(ogr.wkbLineString)
            segment.AddPoint_2D(*point1)
            segment.AddPoint_2D(*point2)
            
            chainage_geom = ogr.Geometry(ogr.wkbPoint)
            chainage_geom.AddPoint_2D(x, y)
            
            distance = chainage_geom.Distance(segment)
            
            if distance < closest_distance:
                closest_distance = distance
                closest_segment_index = i  
                direction_vector = normalized_temp_direction
        
        #print(f"Chainage point at {x}, {y} with chainage {chainage} is closest to segment {closest_segment_index} with direction {direction_vector}")
        directions.append(((x, y), direction_vector, chainage))

    
    return directions

def create_perpendicular_lines_at_chainage(directions, length):
    """
    Create perpendicular lines at specified chainage points along the stream linestring,
    given their direction vectors, and return their start and end coordinates along with the chainage.
    
    :param directions: A list of tuples, each containing a chainage point (tuple of X, Y coordinates) 
                       and its normalized direction vector (tuple of dx, dy).
    :param length: Length of the perpendicular line.
    :return: A list of tuples, each containing the start coordinates (x_start, y_start),
             end coordinates (x_end, y_end), and the chainage of the perpendicular lines created.
    """
    perpendicular_lines = []

    for (x, y), (dx, dy), chainage in directions:
        #dx, dy = direction_vector  

        # Normalize the perpendicular direction vector
        norm = np.sqrt(dx**2 + dy**2)
        dx, dy = dx / norm, dy / norm
        #perp_dx, perp_dy = perp_dx / norm, perp_dy / norm

        # Calculate perpendicular direction by rotating the direction vector 90 degrees
        perp_dx, perp_dy = -dy, dx  # This can be clockwise or counterclockwise


        # Calculate start and end points of the perpendicular line, centered at the chainage point
        x_start = x + perp_dx * (length / 2)
        y_start = y + perp_dy * (length / 2)
        x_end = x - perp_dx * (length / 2)
        y_end = y - perp_dy * (length / 2)

        # Append the start and end coordinates along with the chainage to the list
        perpendicular_lines.append((x_start, y_start, x_end, y_end, chainage))

    return perpendicular_lines


def process_linestring(stream_linestring, points):
    """
    Process a single LineString: Calculate directions and create perpendicular lines.
    """
    directions = calculate_directions_for_chainage_points(stream_linestring, points)
    perpendicular_lines = create_perpendicular_lines_at_chainage(directions, 40)

    return perpendicular_lines


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('input_rasters', nargs='+', type=str, help= 'input DEM raster datasets to sample')
    #argument_parser.add_argument('input_raster', type=str, help= 'input DEM raster dataset to sample')
    argument_parser.add_argument('input_line', type=str, help= 'input line-object vector data source')
    argument_parser.add_argument('input_point', type=str, help= 'input point vector data source')
    argument_parser.add_argument('output_lines', type=str, help='output geometry file')

    input_arguments = argument_parser.parse_args()

    input_raster_path = input_arguments.input_rasters
    input_lines_path = input_arguments.input_line
    input_point_path = input_arguments.input_point
    output_lines_path = input_arguments.output_lines

    vrt_options = gdal.BuildVRTOptions(resampleAlg='bilinear')
    input_raster_vrt = gdal.BuildVRT("/vsimem/input_raster.vrt", input_raster_path, options=vrt_options)
    
    input_raster_dataset = input_raster_vrt 

    input_lines_datasrc = ogr.Open(input_lines_path)
    input_lines_layer = input_lines_datasrc.GetLayer()

    # Load point features
    input_point_datasrc = ogr.Open(input_point_path)
    input_point_layer = input_point_datasrc.GetLayer()

    points = []
    for feature in input_point_layer:
        point_geometry = feature.GetGeometryRef().Clone()  
        chainage = feature.GetField("chainage")  
        points.append((point_geometry,chainage))

    output_lines_driver = ogr.GetDriverByName("gpkg")
    output_lines_datasrc = output_lines_driver.CreateDataSource(output_lines_path)
    output_lines_datasrc.CreateLayer(
        "rendered_lines",
        srs=input_lines_layer.GetSpatialRef(),
        geom_type=ogr.wkbLineString25D,
    )
    output_lines_layer = output_lines_datasrc.GetLayer()

    all_perpendicular_lines = []

    for input_lines_feature in input_lines_layer:
        geom = input_lines_feature.GetGeometryRef()

        if geom.GetGeometryType() == ogr.wkbMultiLineString:
            for i in range(geom.GetGeometryCount()):
                single_linestring = geom.GetGeometryRef(i)
                perpendicular_lines = process_linestring(single_linestring, points)
                all_perpendicular_lines.extend(perpendicular_lines)
        elif geom.GetGeometryType() == ogr.wkbLineString:
            perpendicular_lines = process_linestring(geom, points)
            all_perpendicular_lines.extend(perpendicular_lines)

    # Prepare data storage
    all_lines_data = {
        'line_ids': [],
        'x_coords': [],
        'y_coords': [],
        'z_values': [],
        'distances': [],
    }
    
    for perp_line in all_perpendicular_lines:
        x_start, y_start, x_end, y_end, perp_line_station = perp_line     
     
        # Create an array of x and y coordinates along the line
        x_coords = np.linspace(x_start, x_end, num=50)
        y_coords = np.linspace(y_start, y_end, num=50)

        # Create bounding box encompassing the entire line
        input_line_bbox = BoundingBox(
            x_min=min(x_coords),
            x_max=max(x_coords),
            y_min=min(y_coords),
            y_max=max(y_coords)
        )
        #breakpoint()

        # Get a raster window just covering this line object
        window_raster_dataset = get_raster_window(input_raster_dataset, input_line_bbox)

        # Prepare the interpolator
        window_raster_interpolator = get_raster_interpolator(window_raster_dataset)

        # Interpolate z values along the line
        z_values = window_raster_interpolator((x_coords, y_coords))
        
        
        # Create a new line geometry, including z values
        line_geometry = ogr.Geometry(ogr.wkbLineString25D)
        for x, y, z in zip(x_coords, y_coords, z_values):
            line_geometry.AddPoint(x, y, z)
        
        # Calculate the distances for each row
        distances = [0]  # Initialize with the first distance as 0
        for i in range(1, len(x_coords)):
            dx = x_coords[i] - x_coords[i-1]
            dy = y_coords[i] - y_coords[i-1]
            distance = np.sqrt(dx**2 + dy**2) + distances[-1]
            distances.append(distance)

        feature = ogr.Feature(output_lines_layer.GetLayerDefn())
        feature.SetGeometry(line_geometry)
        output_lines_layer.CreateFeature(feature)
        feature = None

        # Store the line data
        line_id = f'segment_{perp_line_station}'
        all_lines_data['line_ids'].append([line_id] * len(x_coords))
        all_lines_data['x_coords'].append(x_coords)
        all_lines_data['y_coords'].append(y_coords)
        all_lines_data['z_values'].append(z_values)
        all_lines_data['distances'].append(distances)

    concatenated_data = {}
    for key in all_lines_data:
        concatenated_data[key] = np.concatenate(all_lines_data[key])

    # Save to .npz
    output_file_path = r"C:\projekter\rivertopo\tests\data\vejle.npz"

    np.savez_compressed(output_file_path, **concatenated_data)

    ##### Links til inspi #####
    # https://stackoverflow.com/questions/62283718/how-to-extract-a-profile-of-value-from-a-raster-along-a-given-line
    # https://gis.stackexchange.com/questions/167372/extracting-values-from-raster-under-line-using-gdal
    # https://kokoalberti.com/articles/creating-elevation-profiles-with-gdal-and-two-point-equidistant-projection/
    # https://gis.stackexchange.com/questions/50108/elevation-profile-10-km-each-side-of-line
    # https://stackoverflow.com/questions/59144464/plotting-two-cross-section-intensity-at-the-same-time-in-one-figure


if __name__ == '__main__':
    main()
