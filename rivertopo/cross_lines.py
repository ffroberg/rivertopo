#from sampling import BoundingBox, get_raster_window, get_raster_interpolator

from itertools import groupby
from osgeo import gdal, ogr
import numpy as np
from tqdm import tqdm
import argparse
import logging
import random

from profile import RegulativProfilSimpel, RegulativProfilSammensat, OpmaaltProfil  # import interpolation classes

gdal.UseExceptions()
ogr.UseExceptions()

# the method for creating the perpendicular lines was from here: https://gis.stackexchange.com/questions/50108/elevation-profile-10-km-each-side-of-line
def create_perpendicular_line(point1_geometry, point2_geometry, z1, z2):
    x1, y1 = point1_geometry.GetX(), point1_geometry.GetY()
    x2, y2 = point2_geometry.GetX(), point2_geometry.GetY()

    # Calculate the displacement vector for the original line
    vec = np.array([[x2 - x1,], [y2 - y1,]])

    # Rotate the vector 90 deg clockwise and 90 deg counter clockwise
    rot_anti = np.array([[0, -1], [1, 0]])
    rot_clock = np.array([[0, 1], [-1, 0]])
    vec_anti = np.dot(rot_anti, vec)
    vec_clock = np.dot(rot_clock, vec)

    # Normalize the perpendicular vectors
    len_anti = ((vec_anti**2).sum())**0.5
    vec_anti = vec_anti/len_anti
    len_clock = ((vec_clock**2).sum())**0.5
    vec_clock = vec_clock/len_clock

    #unsure about how to determine the lenght of the lines
    length = 10

    # Calculate the coordinates of the endpoints of the perpendicular line
    x3 = x1 + vec_anti[0][0] * length / 2
    y3 = y1 + vec_anti[1][0] * length / 2
    x4 = x1 + vec_clock[0][0] * length / 2
    y4 = y1 + vec_clock[1][0] * length / 2

    # Create the perpendicular line
    line_geometry = ogr.Geometry(ogr.wkbLineString25D)
    line_geometry.AddPoint(x3, y3, float(z1))
    line_geometry.AddPoint(x4, y4, float(z2))

    return line_geometry

def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('input_points_simpel', type=str, help='input points vector data source for simpel')
    argument_parser.add_argument('input_points_sammensat', type=str, help='input points vector data source for sammensat')
    #argument_parser.add_argument('input_points_opmaalt', type=str, help= 'input points vector data source for opmaalt')
    argument_parser.add_argument('output_lines', type=str, help='output geometry file for lines with Z')
    input_arguments = argument_parser.parse_args()

    input_points_simpel_path = input_arguments.input_points_simpel
    input_points_sammensat_path = input_arguments.input_points_sammensat
    #input_points_opmaalt_path = input_arguments.input_points_opmaalt
    output_lines_path = input_arguments.output_lines

    # Load all points from all files into memory and sort by "id" attribute
    points = []
    for input_points_path, profile_type in [(input_points_simpel_path, 'simpel'), (input_points_sammensat_path, 'sammensat')]: #,(input_points_opmaalt_path, 'opmaalt')
        input_points_datasrc = ogr.Open(input_points_path)
        input_points_layer = input_points_datasrc.GetLayer()

        for point_feature in input_points_layer:
            point_feature.profile_type = profile_type  # Add the profile type to the point feature
            points.append(point_feature)

    points.sort(key=lambda f: (f.GetField("laengdeprofillokalid"), f.GetField("regulativstationering")))

    # Now we group the points based on their ID
    grouped_points = {k: sorted(g, key=lambda f: f.GetField("regulativstationering")) for k, g in groupby(points, key=lambda f: f.GetField("laengdeprofillokalid"))}
    
  

    # Create the output file
    output_lines_driver = ogr.GetDriverByName("gpkg")
    output_lines_datasrc = output_lines_driver.CreateDataSource(output_lines_path)
    output_lines_datasrc.CreateLayer(
        "rendered_lines",
        srs=input_points_layer.GetSpatialRef(),
        geom_type=ogr.wkbLineString25D,
    )
    output_lines_layer = output_lines_datasrc.GetLayer()
 
    # Iterate over each group of points and create a line for each pair of points within the group
    for point_group in grouped_points.values():
        for i in range(len(point_group) - 1):
            point1 = point_group[i]
            point2 = point_group[i + 1]

            # Determine which interpolation method to be used for the profile type
         
            if point1.profile_type == 'simpel':
                profile1 = RegulativProfilSimpel(point1)
            elif point1.profile_type == 'sammensat':
                profile1 = RegulativProfilSammensat(point1)
            #elif point1.profile_type == 'opmaalt':
                #profile1 = OpmaaltProfil(point1)

            if point2.profile_type == 'simpel':
                profile2 = RegulativProfilSimpel(point2)
            elif point2.profile_type == 'sammensat':
                profile2 = RegulativProfilSammensat(point2)
            #elif point2.profile_type == 'opmaalt':
                #profile2 = OpmaaltProfil(point2)

            point1_geometry = point1.GetGeometryRef()
            point2_geometry = point2.GetGeometryRef()

            #-----------------------------------------------------------------#
            #Create perpendicular lines and get x and y coordinatoes for cross section data
            x1, y1 = point1_geometry.GetX(), point1_geometry.GetY()
            x2, y2 = point2_geometry.GetX(), point2_geometry.GetY()

            # Calculate the displacement vector for the original line
            vec = np.array([[x2 - x1,], [y2 - y1,]])

            # Rotate the vector 90 deg clockwise and 90 deg counter clockwise
            rot_anti = np.array([[0, -1], [1, 0]])
            rot_clock = np.array([[0, 1], [-1, 0]])
            vec_anti = np.dot(rot_anti, vec)
            vec_clock = np.dot(rot_clock, vec)

            # Normalize the perpendicular vectors
            len_anti = ((vec_anti**2).sum())**0.5
            vec_anti = vec_anti/len_anti
            len_clock = ((vec_clock**2).sum())**0.5
            vec_clock = vec_clock/len_clock

            #unsure about how to determine the lenght of the lines
            length = 10

            # Calculate the coordinates of the endpoints of the perpendicular line
            x3 = x1 + vec_anti[0][0] * length / 2
            y3 = y1 + vec_anti[1][0] * length / 2
            x4 = x1 + vec_clock[0][0] * length / 2
            y4 = y1 + vec_clock[1][0] * length / 2
            #-----------------------------------------------------------------#
            #create x,y values for the crossing line
            t = np.linspace(0,1)
            x = t* (x4-x3)+x3
            y = t* (y4-y3)+y3

            #find the middle
            offset = np.sqrt((x-x3)**2+ (y-y3)**2) - np.sqrt((x1-x3)**2+(y1-y3)**2)

            # Interpolate Z values from points based on the class
            z1_values = profile1.interp(offset)  # returns an array of Z values
            z2_values = profile2.interp(offset)  # returns an array of Z values

            # Create a LineString from the two points
            #line_geometry = ogr.Geometry(ogr.wkbLineString25D)
            #line_geometry.AddPoint(point1_geometry.GetX(), point1_geometry.GetY())
            #line_geometry.AddPoint(point2_geometry.GetX(), point2_geometry.GetY())

            # Create a LineString from the interpolated points
            line_geometry = ogr.Geometry(ogr.wkbLineString25D)

            for i in range(len(t)):
                x_curr = t[i] * (x4-x3) + x3
                y_curr = t[i] * (y4-y3) + y3
                z_curr = (1-t[i]) * z1_values[i] + t[i] * z2_values[i]
                line_geometry.AddPoint(x_curr, y_curr, float(z_curr))
            
            # Create output feature
            output_line_feature = ogr.Feature(output_lines_layer.GetLayerDefn())
            output_line_feature.SetGeometry(line_geometry)
            output_lines_layer.CreateFeature(output_line_feature)

            # create lines for burning into the DEM
            # Define the desired spacing between interpolated points
            spacing = 100.0  # adjust as needed

            # Calculate the distance between the current pair of points
            distance = point1.GetGeometryRef().Distance(point2.GetGeometryRef())

            # Calculate the number of additional points to interpolate between the current pair of points
            num_interpolated_points = int(distance / spacing)
                
            # Interpolate the additional points and create a LineString between each consecutive pair of points
            for j in range(num_interpolated_points + 1):
                # Calculate the factor to determine the position of the current interpolated point
                factor = j / num_interpolated_points if num_interpolated_points != 0 else 1

                # Calculate the coordinates of the current interpolated point
                x_curr = point1.GetGeometryRef().GetX() * (1 - factor) + point2.GetGeometryRef().GetX() * factor
                y_curr = point1.GetGeometryRef().GetY() * (1 - factor) + point2.GetGeometryRef().GetY() * factor
                z_curr = profile1.interp((1 - factor) * point1.GetField('bundkote')) + profile2.interp(factor * point2.GetField('bundkote'))

                # If this is not the first interpolated point, create a LineString between the current interpolated point and the previous one
                if j > 0:
                    # Create a LineString from the interpolated points
                    line_geometry = ogr.Geometry(ogr.wkbLineString25D)
                    line_geometry.AddPoint(x_prev, y_prev, float(z_prev))
                    line_geometry.AddPoint(x_curr, y_curr, float(z_curr))

                    # Create output feature
                    output_line_feature = ogr.Feature(output_lines_layer.GetLayerDefn())
                    output_line_feature.SetGeometry(line_geometry)
                    output_lines_layer.CreateFeature(output_line_feature)

                # Store the coordinates of the current interpolated point for the next iteration
                x_prev, y_prev, z_prev = x_curr, y_curr, z_curr


            
            # Create a perpendicular line at the midpoint of the original line
            #perpendicular_line_geometry = create_perpendicular_line(point1_geometry, point2_geometry, float(z1), float(z2))  
            # output_perpendicular_line_feature = ogr.Feature(output_lines_layer.GetLayerDefn())
            # output_perpendicular_line_feature.SetGeometry(perpendicular_line_geometry)
            # output_lines_layer.CreateFeature(output_perpendicular_line_feature)

    logging.info(f"processed {len(points)} points and created lines")

if __name__ == '__main__':
    main()

