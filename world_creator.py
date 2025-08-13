#
### Import Modules. ###
#
import random
#
from math import floor, ceil, pi, cos, sin, sqrt
#
import numpy as np
from numpy.typing import NDArray
#
from tqdm import tqdm
#
import lib_points as lp
import lib_display as ld  # type: ignore


#
### Point explanation:                                          ###
###     - x = x                                                 ###
###     - y = y                                                 ###
###     - param1 = used for points generation, point weight.    ###
###     - param2 = is a border point or not. (0=false, 1=true)  ###
###     - param3 = terrain elevation (in meters).               ###
###     - param4 = is a river points. (0=false, 1=true)         ###
###     - param5 = forest density. (tree per km²)               ###
###     - param6 = terrain type.                                ###
###     - param7 = city id (default 0 = no city).               ###
#


#
terrain_types: dict[int, str] = {
    0: "plain",
    1: "beach",
    2: "hill",
    3: "mountain",
    4: "high mountain",
    5: "volcano",
    6: "ice mountains",
    7: "ice plain",
    8: "swamp",
    9: "desert",
}


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
        nb_base_boundary_points: int,   # for shapes & random walk
        nb_int_boundary_points: int,    # for random walk
        nb_added_boundary_points: int,  # for shapes & random walk
    ) -> None:

        #
        self.type: str = type
        self.shape: str = shape
        self.center_position_x: int = center_position_x
        self.center_position_y: int = center_position_y
        self.size: float = size
        self.nb_base_boundary_points: int = nb_base_boundary_points
        self.nb_int_boundary_points: int = nb_int_boundary_points
        self.nb_added_boundary_points: int = nb_added_boundary_points
        #
        self.nb_boundary_points: int = nb_base_boundary_points + nb_added_boundary_points


#
def convert_list_points_large_points_area(pts: list[lp.Point]) -> lp.LargePointsAreas:

    #
    pts_area: lp.LargePointsAreas = lp.LargePointsAreas()

    #
    for p in pts:

        #
        pts_area.append( p )

    #
    return pts_area


#
def generate_continent_points(
        tx: int,
        ty: int,
        continent_polygon: lp.Polygon,
        continent_data: InitialContinentData,
        nb_initial_points: int = 10,
        initial_points_strength: int = 10,
        nb_per_points_generated: int = 10,
        continent_superficy_min: int = 10,
        continent_superficy_max: int = 100,
        dist_between_points: int = 40,
    ) -> list[ lp.Point ]:

    #
    points: list[ lp.Point ] = []

    #
    ###
    #
    dsm: int = continent_superficy_max - continent_superficy_min
    #
    dsmf: int = dsm // 4

    #
    continent_superficy: int

    #
    ### Very large continent. ###
    #
    if continent_data.size > tx / 10:

        #
        continent_superficy = random.randint(continent_superficy_min + dsmf * 3, continent_superficy_max)

    #
    ### Large continent. ###
    #
    elif continent_data.size > tx / 30:

        #
        continent_superficy = random.randint(continent_superficy_min + dsmf * 2, continent_superficy_max - dsmf * 1)

    #
    ### Medium sized continent. ###
    #
    elif continent_data.size > tx / 60:

        #
        continent_superficy = random.randint(continent_superficy_min + dsmf * 1, continent_superficy_max - dsmf * 2)

    #
    ### Small continent. ###
    #
    else:

        #
        continent_superficy = random.randint(continent_superficy_min, continent_superficy_max - dsmf * 3)

    #
    base_tots: float = 0

    #
    print(f"Generating {nb_initial_points} initial points...")
    #
    for _ in range(nb_initial_points):

        #
        p: lp.Point = continent_polygon.generate_random_point_uniformly()

        #
        if p is None:

            #
            continue

        #
        p.param1 = initial_points_strength

        #
        points.append( p )

        #
        base_tots += initial_points_strength

    #
    print(f"Generating {continent_superficy} more continent points...")
    #
    for _ in tqdm(range(continent_superficy)):

        #
        base_c_pt_id: int = 0

        #
        base_tire: float = random.uniform(0.0, base_tots)
        #
        crt_base: float = 0
        #
        for base_id, base_pt in enumerate(points):
            #
            crt_base += base_pt.param1
            #
            if base_tire <= crt_base:
                #
                base_c_pt_id = base_id

        #
        mini_point: lp.Point = lp.Point( x=0, y=0 )
        mini_dist: float = float("inf")

        #
        for _ in range(nb_per_points_generated):

            #
            pt: lp.Point = continent_polygon.generate_random_point_uniformly()

            #
            if pt is None:

                #
                continue

            #
            dist: float = pt.calculate_distance( points[base_c_pt_id] )

            #
            if dist < mini_dist:

                #
                mini_point = pt
                mini_dist = dist

        #
        points.append( mini_point )

        #
        points[base_c_pt_id].param1 += 1
        base_tots += 1

    #
    return points


#
def create_continent_random_walk(continent_id: int, initial_continent_data: InitialContinentData, all_points: lp.LargePointsAreas) -> list[lp.Point]:

    #
    points: list[lp.Point] = []

    #
    size: int = int(initial_continent_data.size)

    #
    for _ in range(initial_continent_data.nb_base_boundary_points):

        #
        px: int = initial_continent_data.center_position_x + random.randint(-size, size)
        py: int = initial_continent_data.center_position_y + random.randint(-size, size)

        #
        points.append( lp.Point(x=px, y=py) )

    #
    ### Defensive: if we have too few points, return them as-is. ###
    #
    if len(points) <= 1:
        #
        return points

    #
    ### Compute a robust geometric center (avoid relying on sum(lp.Point)). ###
    #
    cx: float = sum(p.x for p in points) / float(len(points))
    cy: float = sum(p.y for p in points) / float(len(points))
    #
    center: lp.Point = lp.Point(x=round(cx), y=round(cy))

    #
    angle_to_center: list[float] = [
        center.calculate_angle( p ) for p in points
    ]

    #
    indexes: list[int] = list(range(len(angle_to_center)))

    #
    indexes.sort( key = lambda x : angle_to_center[x] )

    #
    ### Build ordered base boundary points (counter-clockwise by angle). ###
    #
    ordered_base: list[lp.Point] = [ points[i] for i in indexes ]

    """
    #
    n: int = len( ordered_base )

    #
    ### Precompute segment distances and allocate intermediate points proportionally. ###
    #
    seg_distances: list[float] = []
    total_perimeter: float = 0.0

    #
    for i in range(n):

        #
        a: lp.Point = ordered_base[i]
        b: lp.Point = ordered_base[(i+1) % n]
        #
        d: float = a.calculate_distance( b )
        #
        seg_distances.append( d )
        #
        total_perimeter += d

    #
    if total_perimeter <= 0.0:
        #
        return ordered_base

    #
    nb_int_total: int = max(0, initial_continent_data.nb_int_boundary_points)

    #
    ### Proportional (floating) allocation then integer rounding with adjustment. ###
    #
    raw_alloc: list[float] = [ (d / total_perimeter) * nb_int_total for d in seg_distances ]
    seg_alloc: list[int] = [ int(round(r)) for r in raw_alloc ]

    #
    ### Fix rounding so sum(seg_alloc) == nb_int_total. ###
    #
    current_sum: int = sum(seg_alloc)
    diff: int = nb_int_total - current_sum

    #
    if diff != 0:

        #
        remainders: list[tuple[int, float]] = [(i, raw_alloc[i] - int(raw_alloc[i])) for i in range(n)]

        #
        ### if we need to add points, prefer largest fractional remainders; if removing, prefer smallest. ###
        #
        remainders.sort(key=lambda x: x[1], reverse=(diff > 0))
        idx: int = 0
        #
        while diff != 0 and idx < n:

            #
            i = remainders[idx][0]

            #
            if diff > 0:
                #
                seg_alloc[i] += 1
                diff -= 1

            #
            else:

                #
                if seg_alloc[i] > 0:

                    #
                    seg_alloc[i] -= 1
                    diff += 1

            #
            idx += 1

    #
    ### Now create a natural constrained random walk for each segment. ###
    #
    boundary: list[lp.Point] = []

    #
    for i in range(n):

        #
        src: lp.Point = ordered_base[i]
        dst: lp.Point = ordered_base[(i+1) % n]

        #
        ### Always keep source (destination will be appended at end of segment processing). ###
        #
        boundary.append( src )

        #
        ### Number of intermediate points to produce on this segment. ###
        #
        k: int = seg_alloc[i]

        #
        if k <= 0:

            #
            ### no interior points for this segment, continue to next (dst will be next segment's src or appended later). ###
            #
            continue

        #
        ### We'll perform k iterative steps that bias to destination while reducing angular variance. ###
        #
        current: lp.Point = src

        #
        for step_index in range(k):

            #
            remaining_steps: int = k - step_index
            remaining_distance: float = current.calculate_distance( dst )

            #
            ### Phase control: early (0..1/3) = noisy, mid = careful, late = tight approach. ###
            #
            frac_done: float = float(step_index) / float(max(1, k))

            #
            if frac_done < 1.0 / 3.0:
                #
                ### allow wide deviation (±pi/2). ###
                #
                max_dev: float = pi / 2.0
                step_len_factor_min: float = 0.4
                step_len_factor_max: float = 1.2

            #
            elif frac_done < 2.0 / 3.0:
                #
                ### moderate deviation (±pi/6). ###
                #
                max_dev: float = pi / 6.0
                step_len_factor_min: float = 0.6
                step_len_factor_max: float = 1.1

            #
            else:
                #
                ### tight deviation near destination (±pi/12). ###
                #
                max_dev: float = pi / 12.0
                step_len_factor_min: float = 0.8
                step_len_factor_max: float = 1.0

            #
            ### Desired direction is toward dst; sample an angle within the cone around that direction. ###
            #
            desired_angle: float = current.calculate_angle( dst )
            #
            angle: float = desired_angle + random.uniform( -max_dev, max_dev )

            #
            ### Choose a step length proportional to remaining distance and remaining steps. ###
            ### Clamp to avoid a single huge jump. ###
            #
            if remaining_steps <= 0:
                #
                chosen_len: float = max(1.0, remaining_distance)
            #
            else:
                #
                avg_needed: float = remaining_distance / float(remaining_steps)
                chosen_len: float = avg_needed * random.uniform(step_len_factor_min, step_len_factor_max)
                #
                ### Prevent overshoot: ensure we leave at least (remaining_steps - 1) * 0.5 distance for subsequent steps. ###
                #
                min_reserve: float = 0.5 * float(max(0, remaining_steps - 1))
                chosen_len = max(1.0, min(chosen_len, max(1.0, remaining_distance - min_reserve)))

            #
            dx: int = round( chosen_len * cos( angle ) )
            dy: int = round( chosen_len * sin( angle ) )

            #
            new_pt: lp.Point = current + lp.Point( x=dx, y=dy )

            #
            ### If we are already very close to destination, snap to it to avoid tiny residuals. ###
            #
            if new_pt.calculate_distance( dst ) <= 1.0:
                #
                new_pt = dst
                boundary.append( new_pt )
                current = new_pt
                #
                break

            #
            boundary.append( new_pt )
            current = new_pt

        #
        ### Ensure the destination is present (avoid duplicate when next segment starts with it). ###
        #
        if boundary[-1].calculate_distance( dst ) > 0.0:
            #
            boundary.append( dst )

    #
    return boundary
    """

    return ordered_base


#
def create_continent_shape_circle(continent_id: int, initial_continent_data: InitialContinentData, all_points: lp.LargePointsAreas) -> list[lp.Point]:

    #
    boundary_points: list[lp.Point] = []

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
    return boundary_points


#
def create_continent_shape_square(continent_id: int, initial_continent_data: InitialContinentData, all_points: lp.LargePointsAreas) -> list[lp.Point]:

    #
    boundary_points: list[lp.Point] = []

    #
    nps: int = ceil( initial_continent_data.nb_base_boundary_points / 4 )

    #
    rad: int = floor( initial_continent_data.size / nps )

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
    return boundary_points


#
def create_continent_shape_ellipse(continent_id: int, initial_continent_data: InitialContinentData, all_points: lp.LargePointsAreas) -> list[lp.Point]:

    #
    boundary_points: list[lp.Point] = []

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
    return boundary_points


#
def add_details_to_polygon_borders(continent_id: int, initial_continent_data: InitialContinentData, all_points: lp.LargePointsAreas, continents_boundary_points: list[lp.Point]) -> lp.PointCluster:

    #
    ### Add interpolation with noise details points for more natural borders. ###
    #
    for _ in range( initial_continent_data.nb_added_boundary_points ):

        #
        n: int = len(continents_boundary_points)

        #
        idx_to_insert: int = random.randint(1, n)

        #
        p1: lp.Point = continents_boundary_points[idx_to_insert-1]
        p2: lp.Point = continents_boundary_points[idx_to_insert%n]

        #
        npx: int = (p1.x + p2.x) / 2
        npy: int = (p1.y + p2.y) / 2
        #
        pd: float = p1.calculate_distance( p2 ) / (5 * sqrt(2))

        #
        dx: int = round( random.uniform(-pd, pd) )
        dy: int = round( random.uniform(-pd, pd) )

        #
        new_point: lp.Point = lp.Point( x = ( npx + dx ) , y = ( npy + dy ) )

        #
        continents_boundary_points.insert( idx_to_insert, new_point )

    #
    nb_pts: int = len(continents_boundary_points)

    #
    ### Populate the PointCluster with the generated boundary points. ###
    #
    continent_boundary: lp.PointCluster = lp.PointCluster(init_size=nb_pts)
    #
    for i in range(nb_pts):
        #
        continent_boundary.data[i] = continents_boundary_points[i].data

    #
    continent_boundary.length = nb_pts

    #
    return continent_boundary


#
def create_continent_polygon(continent_id: int, initial_continent_data: InitialContinentData, all_points: lp.LargePointsAreas) -> lp.Polygon:

    #
    continent_points: list[lp.Point]

    #
    if initial_continent_data.type == "random_walk":

        #
        continent_points = create_continent_random_walk(continent_id=continent_id, initial_continent_data=initial_continent_data, all_points=all_points)

    #
    elif initial_continent_data.type == "shape":

        #
        if initial_continent_data.shape == "circle":

            #
            continent_points = create_continent_shape_circle(continent_id=continent_id, initial_continent_data=initial_continent_data, all_points=all_points)

        #
        elif initial_continent_data.shape == "ellipse":

            #
            continent_points = create_continent_shape_ellipse(continent_id=continent_id, initial_continent_data=initial_continent_data, all_points=all_points)

        #
        elif initial_continent_data.shape == "square":

            #
            continent_points = create_continent_shape_square(continent_id=continent_id, initial_continent_data=initial_continent_data, all_points=all_points)

        #
        else:

            #
            raise UserWarning(f"Error: Unknown initial continent shape : `{initial_continent_data.shape}` !")

    #
    else:

        #
        raise UserWarning(f"Error: Unknown initial continent type : `{initial_continent_data.type}` !")

    #
    continent_polygon_boundary = add_details_to_polygon_borders(continent_id=continent_id, initial_continent_data=initial_continent_data, all_points=all_points, continents_boundary_points=continent_points)

    #
    continent_polygon: lp.Polygon = lp.Polygon(boundary=continent_polygon_boundary, grid_context=all_points)

    #
    return continent_polygon


#
def terrain_generator(
        tx: int = 2048,
        ty: int = 2048,
        initial_continents_attemps: int = 20,
        nb_initial_points: int = 10,
        initial_points_strength: int = 20,
        nb_per_points_generated: int = 40,
        continent_superficy_min: int = 100,
        continent_superficy_max: int = 1000,
        dist_between_points: int = 20,
        initial_continents: list[InitialContinentData] = []
    ) -> None:

    #
    ### -------- STEP 0: generate continent polygons. -------- ###
    #

    #
    all_points_for_heightmap: lp.LargePointsAreas = lp.LargePointsAreas()

    #
    if not initial_continents:

        #
        generated: list[ tuple[float, float, float] ] = []

        #
        print("No initial_continents provided: generating random initial continent descriptors...")
        #
        for _ in range(initial_continents_attemps):

            #
            t: bool = False if random.uniform(0, 1) <= 0.6 else True
            #
            cont_type: str = "shape" if t else "random_walk"

            #
            shape: str = random.choice(["circle", "ellipse"]) if t else "random_walk"

            #
            nbbpts: int = random.randint(4, 10)
            nbipts: int = random.randint(1, 6)
            nbapts: int = random.randint(100, 1000)

            #
            csize: float = random.uniform(tx/100, tx/4)

            #
            min_margin: float = tx / 10

            #
            cpx: int = int( random.uniform(min_margin + csize, tx - min_margin - csize) )
            cpy: int = int( random.uniform(min_margin + csize, ty - min_margin - csize) )

            #
            good: bool = True

            #
            for prev in generated:

                #
                d: float = sqrt( ( cpx - prev[0] ) ** 2 + ( cpy - prev[1] ) ** 2 )

                #
                if d <= csize + prev[2]:

                    #
                    good = False

                    #
                    break

            #
            if not good:

                #
                continue

            #
            initial_continents.append(

                InitialContinentData(
                    type=cont_type,
                    shape=shape,
                    center_position_x=cpx,
                    center_position_y=cpy,
                    size=csize,
                    nb_base_boundary_points=nbbpts,
                    nb_int_boundary_points=nbipts,
                    nb_added_boundary_points=nbapts,
                )

            )

            #
            generated.append( (cpx, cpy, csize) )

    #
    continent_polygons: list[lp.Polygon] = [
        create_continent_polygon( continent_id=i, initial_continent_data=init_continent_data, all_points=all_points_for_heightmap )
        for i, init_continent_data in enumerate(initial_continents)
    ]

    #
    # ld.render_only_polygons(tx=tx, ty=ty, polygons=continent_polygons)


    #
    ### -------- STEP 1: generate continent points. -------- ###
    #

    #
    n_continents: int = len(continent_polygons)

    #
    continent_points: list[ lp.LargePointsAreas ] = [

        convert_list_points_large_points_area(
            generate_continent_points(
                tx=tx,
                ty=ty,
                continent_polygon=continent_polygons[i],
                continent_data=initial_continents[i],
                nb_initial_points=nb_initial_points,
                initial_points_strength=initial_points_strength,
                nb_per_points_generated=nb_per_points_generated,
                continent_superficy_min=continent_superficy_min,
                continent_superficy_max=continent_superficy_max,
                dist_between_points=dist_between_points
            )
        )

        for i in range(n_continents)

    ]

    #
    ld.render_points_with_colors_from_points_areas_with_polygons(
        tx=tx,
        ty=ty,
        point_clusters=continent_points,
        colors=ld.generate_random_colors( n = n_continents ),
        polygons=continent_polygons
    )


    #
    ### -------- STEP 2: generate continent heightmap. -------- ###
    #

    #
    ### Initialize for all the points in the world to prepare to render the heightmap. ###
    #
    all_points_for_heightmap: lp.LargePointsAreas = lp.LargePointsAreas()

    #
    ### For all continents. ###
    #

    #
    c_id: int
    #
    for c_id, cpolygon in enumerate(continent_polygons):

        #
        ### TODO: set param3 the elevation of all the points. ###
        ### TODO: elevation values are in meter. So make plains, beachs low toward 0, hills to 100 ~ 500 meters and mountains to 1k ~ 6k high for instance. ###
        ### TODO: create logical mountains chains, hills and plains and all the other terrain types. ###
        ### TODO: initial polygon points have an elevation of 0 because of default beach type. ###

        #
        ### Border points to 0 of elevation. ###
        #
        p: lp.Point
        #
        for p in cpolygon.boundary:

            #
            p.param3 = 0

            #
            all_points_for_heightmap.append( p )

        #
        ### Inside points to hills, mountains and plains. ###
        #
        for p in continent_points[c_id]:

            # TODO: good conditions to be mountains, hills, plains, beach, forest, desert, city, etc...

            #
            terrain_type: str = random.choice( [
                "very high mountains",
                "high mountains",
                "mid mountains",
                "low mountains",
                "hills",
                "swamps",
                "forest",
                "mountain forest",
                "plains",
                "city",
                "desert",
                "beach",
            ] )


            if terrain_type == "very high mountains":
                #
                p.param3 = random.uniform(4500, 6000)

            if terrain_type == "high mountains":
                #
                p.param3 = random.uniform(2500, 5000)

            if terrain_type == "mid mountains":
                #
                p.param3 = random.uniform(1000, 3000)

            if terrain_type == "low mountains":
                #
                p.param3 = random.uniform(500, 1500)

            if terrain_type == "hills":
                #
                p.param3 = random.uniform(150, 500)

            if terrain_type == "swamps":
                #
                p.param3 = random.uniform(-50, 50)

            if terrain_type == "plains":
                #
                p.param3 = random.uniform(0, 50)

            if terrain_type == "desert":
                #
                p.param3 = random.uniform(0, 50)

            if terrain_type == "beach":
                #
                p.param3 = random.uniform(0, 10)

            #
            all_points_for_heightmap.append( p )


    #
    ### Generate an heightmap texture for all the planet. ###
    #
    arr: NDArray[np.float32] = np.zeros( shape=(tx, ty), dtype=np.float32 )

    #
    previous_in_poly_id: int = -1

    #
    for x in range(tx):

        #
        for y in range(ty):

            #
            crt_p: lp.Point = lp.Point( x=x, y=y )

            #
            in_poly_id: int = -1

            #
            if previous_in_poly_id != -1 and continent_polygons[previous_in_poly_id].__contains__(crt_p):

                #
                in_poly_id = previous_in_poly_id

            #
            else:

                #
                for c_id, cpoly in enumerate( continent_polygons ):

                    #
                    if c_id == previous_in_poly_id:

                        #
                        continue

                    #
                    if cpoly.__contains__(crt_p):

                        #
                        in_poly_id: int = c_id
                        #
                        break

            #
            ### Default: Water. ###
            #
            crt_elevation: float = -1000

            #
            if in_poly_id >= 0:

                #
                pts: list[lp.Point] = all_points_for_heightmap.get_all_points_in_circle( point=lp.Point( x=x, y=y ), radius=tx/10 )

                #
                total_distances: float = 0
                #
                crt_elevation = 0

                #
                for p in pts:

                    #
                    dst: float = p.calculate_distance( crt_p )

                    #
                    total_distances += dst

                    #
                    crt_elevation += p.param3 * dst

                #
                crt_elevation /= total_distances

            #
            arr[x, y] = crt_elevation

    #
    ld.draw_heightmaps( arr=arr )

    #
    ### -------- STEP 3: More natural  -------- ###
    #

    # TODO: Add mountains, rivers, lakes, lands to continents. (rivers must have branches, and etc...). Separate ocean from continents.

    # TODO: Add cities to continents.

    # TODO: Create countries from cities / favorise natural countries borders like mountains chains or rivers.

    # TODO: Create Buildings for each city.

    # TODO: Create people for each city / country.


#
if __name__ == "__main__":

    #
    ld.init_plt()

    #
    terrain_generator()

    #
    ld.quit_plt()

