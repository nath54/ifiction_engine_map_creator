#
### Import Modules. ###
#
import random
#
from math import floor, ceil, pi, cos, sin
#
import numpy as np  # type: ignore
#
from tqdm import tqdm
#
import lib_points as lp
import lib_display as ld  # type: ignore


#
class InitialContinentData:

    #
    def __init__(
        self,
        type: str,
        shape: str,
        center_position_x: int,
        center_position_y: int,
        size: float,
        nb_base_boundary_points: int,
        nb_added_boundary_points: int,
    ) -> None:

        #
        self.type: str = type
        self.shape: str = shape
        self.center_position_x: int = center_position_x
        self.center_position_y: int = center_position_y
        self.size: float = size
        self.nb_base_boundary_points: int = nb_base_boundary_points
        self.nb_added_boundary_points: int = nb_added_boundary_points
        #
        self.nb_boundary_points: int = nb_base_boundary_points + nb_added_boundary_points


#
def generate_continent_points(
        nb_continents: int = 4,
        tx: int = 2048,
        ty: int = 2048,
        continent_superficy_min: int = 10,
        continent_superficy_max: int = 100,
        dist_between_points: int = 40,
        border_margin: int = 300
    ) -> list[ lp.Point ]:

    #
    points: list[ lp.Point ] = []

    #
    for _continent_id in range(nb_continents):

        #
        print(f"Continent : {_continent_id + 1} / {nb_continents}")

        #
        continent_center_x = random.randint(border_margin, tx-1-border_margin)
        continent_center_y = random.randint(border_margin, ty-1-border_margin)

        #
        continent_superficy = random.randint(continent_superficy_min, continent_superficy_max)

        #
        base_tots: int = 1

        #
        points.append( lp.Point(x=continent_center_x, y=continent_center_y, params=[1, 0]) )

        #
        for _ in tqdm(range(continent_superficy)):

            #
            base_c_pt_id: int = 0

            #
            base_tire: int = random.randint(0, base_tots)
            #
            crt_base: int = 0
            #
            for base_id, base_pt in enumerate(points):
                #
                crt_base += base_pt.param1
                #
                if base_tire <= crt_base:
                    #
                    base_c_pt_id = base_id

            #
            points[base_c_pt_id].param1 += 1
            base_tots += 1

            #
            new_px: int = max(0, min(tx, points[base_c_pt_id].x + random.randint(-dist_between_points, dist_between_points)))
            new_py: int = max(0, min(ty, points[base_c_pt_id].y + random.randint(-dist_between_points, dist_between_points)))

            #
            ### Reduce the probability of inside border margin. ###
            #
            if new_px < border_margin or new_px > tx - border_margin or new_py < border_margin or new_py > ty - border_margin:

                #
                new_px = max(0, min(tx, points[base_c_pt_id].x + random.randint(-dist_between_points, dist_between_points)))
                new_py = max(0, min(ty, points[base_c_pt_id].y + random.randint(-dist_between_points, dist_between_points)))

            #
            points.append( lp.Point(x=new_px, y=new_py, params=[0, 0]) )

    #
    return points


#
def create_cluster_of_points(
        points: list[ lp.Point ],
        nb_continents: int,
        treshold_point_continent_distance: float = 100
    ) -> list[ lp.LargePointsAreas ]:

    #
    continents_points: list[ lp.LargePointsAreas ] = []

    #
    print(f"Distribute all the {len(points)} points in continents.")
    #
    p: lp.Point
    #
    for p in tqdm( points ):

        #
        ### Try to find the continent id to add the point to. ###
        #
        continents_distances: list[ tuple[int, float] ] = []

        #
        cid: int
        cpts: lp.LargePointsAreas
        #
        for cid, cpts in enumerate( continents_points ):

            #
            dist_to_continent: float = cpts.distance_from_point( point=p )

            #
            if dist_to_continent < treshold_point_continent_distance:

                #
                continents_distances.append( (cid, dist_to_continent) )

        #
        ### If not continents close enough from current point. ###
        #
        if len( continents_distances ) == 0:

            #
            ### Create a new cluster of points for a new continent. ###
            #
            continents_points.append(
                lp.LargePointsAreas()
            )

            #
            continents_points[-1].append( value=p )

        #
        ### If ONE continent has been found. ###
        #
        elif len(continents_distances) == 1:

            #
            continents_points[continents_distances[0][0]].append( value=p )

        #
        ### MULTIPLE continents to merge. ###
        #
        else:

            #
            ### Sort the continents cluster index to inverse growing order so we will be able to pop without shifting. ###
            #
            continents_distances.sort( key = lambda x: x[0], reverse=True )

            #
            ### MERGE all the continents to merge. ###
            #
            new_continent: lp.LargePointsAreas = continents_points[continents_distances[0][0]].__add__( continents_points[continents_distances[1][0]] )

            #
            for cdts in continents_distances[2:]:

                #
                new_continent = new_continent.__add__( continents_points[cdts[0]] )

            #
            ### Removing all the previous continents. ###
            #
            for cdts in continents_distances:

                #
                continents_points.pop( cdts[0] )

            #
            ### Adding the new final continent. ###
            #
            continents_points.append( new_continent )

    #
    ### . ###
    #
    return continents_points


#
def create_continent_random_walk(continent_id: int, initial_continent_data: InitialContinentData, all_points: lp.LargePointsAreas) -> lp.PointCluster:

    #
    continent_boundary: lp.PointCluster = lp.PointCluster(init_size=initial_continent_data.nb_boundary_points)

    # TODO: random walk

    #
    return continent_boundary


#
def create_continent_shape_circle(continent_id: int, initial_continent_data: InitialContinentData, all_points: lp.LargePointsAreas) -> lp.PointCluster:

    #
    boundary_points: list[lp.Point] = []

    #
    dd: int = 2 * floor( initial_continent_data.size / initial_continent_data.nb_added_boundary_points )

    #
    nb_added_pts_per_pts: int = floor( initial_continent_data.nb_added_boundary_points / initial_continent_data.nb_base_boundary_points )

    #
    nb_remaining_added_pts: int = initial_continent_data.nb_added_boundary_points

    #
    r: float = initial_continent_data.size

    #
    da: float = ( 2 * pi ) / initial_continent_data.nb_base_boundary_points

    #
    a: float = ( pi - da ) / 2

    #
    rad: float = r * cos( a )

    #
    for i in range(initial_continent_data.nb_base_boundary_points):

        #
        crt_agl: float = i * da

        #
        cx: int = initial_continent_data.center_position_x + round( r * cos(crt_agl) )
        cy: int = initial_continent_data.center_position_y + round( r * sin(crt_agl) )

        #
        cagl: float = random.uniform( 0, 2 * pi )

        #
        crad: float = random.uniform( 0, rad )

        #
        rpx: int = cx + round( crad * cos(cagl) )
        rpy: int = cy + round( crad * sin(cagl) )

        #
        boundary_points.append( lp.Point(x=rpx, y=rpy) )

        #
        if i != 0 and nb_remaining_added_pts > 0:

            #
            for _ in range(0, min(nb_added_pts_per_pts, nb_remaining_added_pts)):

                #
                dx: int = random.randint( -dd, dd )
                dy: int = random.randint( -dd, dd )

                #
                dpt: lp.Point = lp.Point( x=dx, y=dy )

                #
                boundary_points.insert( -2, dpt + ( boundary_points[-1] + boundary_points[-2] ) / 2 )

            #
            nb_remaining_added_pts -= nb_added_pts_per_pts

    #
    continent_boundary: lp.PointCluster = lp.PointCluster(init_size=initial_continent_data.nb_boundary_points)

    #
    for i in range(initial_continent_data.nb_boundary_points):

        #
        continent_boundary.data[i] = boundary_points[i].data

    #
    return continent_boundary


#
def create_continent_shape_square(continent_id: int, initial_continent_data: InitialContinentData, all_points: lp.LargePointsAreas) -> lp.PointCluster:

    #
    boundary_points: list[lp.Point] = []

    #
    nps: int = ceil( initial_continent_data.nb_base_boundary_points / 4 )

    #
    rad: int = floor( initial_continent_data.size / nps )

    #
    dd: int = 2 * floor( initial_continent_data.size / initial_continent_data.nb_boundary_points )

    #
    nb_added_pts_per_pts: int = floor( initial_continent_data.nb_added_boundary_points / initial_continent_data.nb_base_boundary_points )

    #
    nb_remaining_added_pts: int = initial_continent_data.nb_added_boundary_points

    #
    x1: int = round( initial_continent_data.center_position_x - initial_continent_data.size )
    x2: int = round( initial_continent_data.center_position_x + initial_continent_data.size )
    x3: int = x2
    x4: int = x1

    #
    y1: int = round( initial_continent_data.center_position_y - initial_continent_data.size )
    y2: int = y1
    y3: int = round( initial_continent_data.center_position_y + initial_continent_data.size )
    y4: int = y3

    #
    for i in range(initial_continent_data.nb_base_boundary_points):

        #
        cx: int
        cy: int

        #
        if i // nps == 0:

            #
            cx = x1 + rad * 2 * i
            cy = y1

        #
        elif i // nps == 1:

            #
            cx = x1
            cy = y2 + rad * 2 * (i - nps)

        #
        elif i // nps == 2:

            #
            cx = x3 - rad * 2 * (i - 2 * nps)
            cy = y3

        #
        else:

            #
            cx = x4
            cy = y4 - rad * 2 * (i - 3 * nps)

        #
        cagl: float = random.uniform( 0, 2 * pi )

        #
        crad: float = random.uniform( 0, rad )

        #
        rpx: int = cx + round( crad * cos(cagl) )
        rpy: int = cy + round( crad * sin(cagl) )

        #
        boundary_points.append( lp.Point(x=rpx, y=rpy) )

        #
        if i != 0 and nb_remaining_added_pts > 0:

            #
            for _ in range(0, min(nb_added_pts_per_pts, nb_remaining_added_pts)):

                #
                dx: int = random.randint( -dd, dd )
                dy: int = random.randint( -dd, dd )

                #
                dpt: lp.Point = lp.Point( x=dx, y=dy )

                #
                boundary_points.insert( -2, dpt + ( boundary_points[-1] + boundary_points[-2] ) / 2 )

            #
            nb_remaining_added_pts -= nb_added_pts_per_pts

    #
    continent_boundary: lp.PointCluster = lp.PointCluster(init_size=initial_continent_data.nb_boundary_points)

    #
    for i in range(initial_continent_data.nb_boundary_points):

        #
        continent_boundary.data[i] = boundary_points[i].data

    #
    return continent_boundary


#
def create_continent_shape_ellipse(continent_id: int, initial_continent_data: InitialContinentData, all_points: lp.LargePointsAreas) -> lp.PointCluster:

    #
    boundary_points: list[lp.Point] = []

    #
    dd: int = 2 * floor( initial_continent_data.size / initial_continent_data.nb_added_boundary_points )

    #
    nb_added_pts_per_pts: int = floor( initial_continent_data.nb_added_boundary_points / initial_continent_data.nb_base_boundary_points )

    #
    nb_remaining_added_pts: int = initial_continent_data.nb_added_boundary_points

    #
    ### Define ellipse radii and a random rotation angle. ###
    #
    radius_x: float = initial_continent_data.size
    radius_y: float = initial_continent_data.size / 2
    rotation_angle: float = random.uniform( 0, 2 * pi )

    #
    ### Calculate the angular step between base points. ###
    #
    da: float = ( 2 * pi ) / initial_continent_data.nb_base_boundary_points

    #
    ### The radius for the random displacement of base points. ###
    #
    rad: float = initial_continent_data.size / initial_continent_data.nb_base_boundary_points

    #
    for i in range(initial_continent_data.nb_base_boundary_points):

        #
        ### Calculate the point on the ellipse for the current angle. ###
        #
        angle: float = i * da
        #
        unrotated_x: float = radius_x * cos(angle)
        unrotated_y: float = radius_y * sin(angle)

        #
        ### Apply the rotation to the point. ###
        #
        rotated_x: float = unrotated_x * cos(rotation_angle) - unrotated_y * sin(rotation_angle)
        rotated_y: float = unrotated_x * sin(rotation_angle) + unrotated_y * cos(rotation_angle)

        #
        ### Translate the point to the continent's center. ###
        #
        cx: int = round( initial_continent_data.center_position_x + rotated_x )
        cy: int = round( initial_continent_data.center_position_y + rotated_y )

        #
        ### Add a random displacement to the point for a more natural look. ###
        #
        cagl: float = random.uniform( 0, 2 * pi )
        crad: float = random.uniform( 0, rad )
        #
        rpx: int = cx + round( crad * cos(cagl) )
        rpy: int = cy + round( crad * sin(cagl) )

        #
        boundary_points.append( lp.Point(x=rpx, y=rpy) )

        #
        ### Add intermediate points between the last two base points to roughen the coastline. ###
        #
        if i != 0 and nb_remaining_added_pts > 0:

            #
            for _ in range(0, min(nb_added_pts_per_pts, nb_remaining_added_pts)):

                #
                dx: int = random.randint( -dd, dd )
                dy: int = random.randint( -dd, dd )

                #
                dpt: lp.Point = lp.Point( x=dx, y=dy )

                #
                ### Insert a new point near the midpoint of the last two created points. ###
                #
                boundary_points.insert( -2, dpt + ( boundary_points[-1] + boundary_points[-2] ) / 2 )

            #
            nb_remaining_added_pts -= nb_added_pts_per_pts

    #
    ### Populate the PointCluster with the generated boundary points. ###
    #
    continent_boundary: lp.PointCluster = lp.PointCluster(init_size=initial_continent_data.nb_boundary_points)
    #
    for i in range(initial_continent_data.nb_boundary_points):
        #
        continent_boundary.data[i] = boundary_points[i].data

    #
    return continent_boundary


#
def create_continent_polygon(continent_id: int, initial_continent_data: InitialContinentData, all_points: lp.LargePointsAreas) -> lp.Polygon:

    #
    continent_polygon_boundary: lp.PointCluster

    #
    if initial_continent_data.type == "random_walk":

        #
        continent_polygon_boundary = create_continent_random_walk(continent_id=continent_id, initial_continent_data=initial_continent_data, all_points=all_points)

    #
    elif initial_continent_data.type == "shape":

        #
        if initial_continent_data.shape == "circle":

            #
            continent_polygon_boundary = create_continent_shape_circle(continent_id=continent_id, initial_continent_data=initial_continent_data, all_points=all_points)

        #
        elif initial_continent_data.shape == "ellipse":

            #
            continent_polygon_boundary = create_continent_shape_ellipse(continent_id=continent_id, initial_continent_data=initial_continent_data, all_points=all_points)

        #
        elif initial_continent_data.shape == "square":

            #
            continent_polygon_boundary = create_continent_shape_square(continent_id=continent_id, initial_continent_data=initial_continent_data, all_points=all_points)

        #
        else:

            #
            raise UserWarning(f"Error: Unknown initial continent shape : `{initial_continent_data.shape}` !")

    #
    else:

        #
        raise UserWarning(f"Error: Unknown initial continent type : `{initial_continent_data.type}` !")

    #
    continent_polygon: lp.Polygon = lp.Polygon(boundary=continent_polygon_boundary, grid_context=all_points)

    #
    return continent_polygon


#
def terrain_generator(
        tx: int = 2048,
        ty: int = 2048,
        nb_continents: int = 4,
        continent_superficy_min: int = 100,
        continent_superficy_max: int = 1000,
        dist_between_points: int = 20,
        treshold_point_continent_distance: float = 40,
        border_margin: int = 300,
        radius_border_points: float = 60,
        dead_angle_min_border_points: float = 100,
        radius_between_border_points: float = 1,
        search_radius_factor: float = 0.2,
        initial_continents: list[InitialContinentData] = []
    ) -> None:

    pass

    # TODO: Generate continent polygons

    # TODO: Generate continent points inside continents polygons

    # TODO: Add mountains, rivers, lakes, lands to continents. (rivers must have branches, and etc...). Separate ocean from continents.

    # TODO: Add cities to continents.

    # TODO: Create countries from cities / favorise natural countries borders like mountains chains or rivers.

    # TODO: Create Buildings for each city.

    # TODO: Create people for each city / country.


#
if __name__ == "__main__":

    #
    terrain_generator()
