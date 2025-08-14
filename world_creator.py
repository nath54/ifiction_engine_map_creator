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
### Point explanation:                                                      ###
###     - x = x                                                             ###
###     - y = y                                                             ###
###     - param1 = used for points generation, point weight.                ###
###     - param2 = is a border point or not. (0=false, 1=true)              ###
###     - param3 = terrain elevation (in meters).                           ###
###     - param4 = river size ( <= 0 if no rivers, else siez in meters).    ###
###     - param5 = city id (default 0 = no city).                           ###
#


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
def create_heightmap_points(tx: int, ty: int, continents_datas: list[InitialContinentData], continent_polygons: list[lp.Polygon]) -> tuple[lp.LargePointsAreas, list[tuple[lp.Point, lp.Point]]]:

    #
    all_pts_hm: lp.LargePointsAreas = lp.LargePointsAreas(sub_cluster_size=10)

    #
    rivers_segments: list[tuple[lp.Point, lp.Point]] = []

    #
    # Iterate continents and produce: border points, mountain chains, hills/plains and rivers.
    #
    for i in range(len(continent_polygons)):

        #
        c_data: InitialContinentData = continents_datas[i]
        c_poly: lp.Polygon = continent_polygons[i]

        #
        # 1) Add polygon boundary points as mostly beach / gravel.
        #
        for bidx in range(len(c_poly.boundary)):

            #
            bp: lp.Point = c_poly.boundary[bidx]
            hp: lp.Point = lp.Point(x=int(bp.x), y=int(bp.y))

            #
            # param2 = border flag
            #
            hp.param2 = 1

            #
            # Majority are gentle beach heights; a small fraction are small cliffs (negative).
            #
            if random.random() < 0.03:
                #
                hp.param3 = float(random.randint(-80, -8))
            else:
                #
                hp.param3 = float(random.randint(-5, 8))

            #
            hp.param4 = 0.0  # no river here by default
            hp.param1 = 0.0
            hp.param5 = 0.0

            #
            all_pts_hm.append(hp)

        #
        # 2) Mountain chains: generate a few random-walk chains with possible branches.
        #
        mountains_lp: lp.LargePointsAreas = lp.LargePointsAreas(sub_cluster_size=all_pts_hm.sub_cluster_size)

        #
        # Heuristic: number of chains scales with continent size relative to map width.
        #
        size_scale: float = max(0.25, float(c_data.size) / max(1.0, float(tx) / 4.0))
        #
        num_chains: int = max(3, int(round(size_scale * 20.0)))

        #
        for chain_idx in range(num_chains):

            #
            # start inside polygon (use polygon's uniform sampler)
            #
            start_pt: lp.Point = c_poly.generate_random_point_uniformly()
            if start_pt is None:
                #
                continue

            #
            # chain geometry parameters scale with continent size
            #
            chain_length: int = max(5, int(max(20, c_data.size / 10.0)))
            #
            peak_scale: float = size_scale
            #
            peak_height: float = float(int(random.uniform(1200.0, 6000.0) * peak_scale))

            #
            angle: float = random.uniform(0.0, 2.0 * pi)
            current: lp.Point = lp.Point(x=int(start_pt.x), y=int(start_pt.y))

            #
            for step in range(chain_length):

                #
                # step length biased with continent size but clamped
                #
                step_len: float = random.uniform(4.0, max(6.0, c_data.size / 20.0))
                #
                dx: int = int(round(step_len * cos(angle) + random.uniform(-2.0, 2.0)))
                dy: int = int(round(step_len * sin(angle) + random.uniform(-2.0, 2.0)))

                #
                cand: lp.Point = lp.Point(x=int(current.x + dx), y=int(current.y + dy))

                #
                # keep the chain inside the polygon (reflect / try a few times)
                #
                tries: int = 0
                while tries < 6 and not (cand in c_poly):
                    #
                    angle += pi * random.uniform(0.1, 0.9)  # reflect with noise
                    dx = int(round(step_len * cos(angle)))
                    dy = int(round(step_len * sin(angle)))
                    cand = lp.Point(x=int(current.x + dx), y=int(current.y + dy))
                    tries += 1

                #
                if not (cand in c_poly):
                    #
                    # couldn't keep inside after a few tries; stop the chain gracefully.
                    #
                    break

                #
                # elevation profile: parabolic profile peaking near the middle of chain
                #
                t: float = float(step) / float(max(1, chain_length - 1))
                #
                # parabolic falloff (peak at t=0.5)
                #
                profile: float = max(0.0, 1.0 - 4.0 * (t - 0.5) ** 2)
                #
                elev: float = peak_height * profile + random.uniform(-peak_height * 0.08, peak_height * 0.08)
                elev = max(-500.0, elev)

                #
                mpt: lp.Point = lp.Point(x=int(cand.x), y=int(cand.y))
                mpt.param2 = 0.0
                mpt.param3 = float(elev)
                mpt.param4 = 0.0
                mpt.param1 = 0.0
                mpt.param5 = 0.0

                #
                mountains_lp.append(mpt)
                all_pts_hm.append(mpt)

                #
                current = cand
                #
                # slowly change angle (smooth wiggle)
                #
                angle += random.uniform(-pi / 6.0, pi / 6.0)

                #
                # occasional branch
                #
                if random.random() < 0.08 and chain_length >= 6:
                    #
                    branch_len: int = max(3, int(chain_length / 3))
                    branch_angle: float = angle + random.uniform(-pi / 2.0, pi / 2.0)
                    branch_peak: float = peak_height * random.uniform(0.3, 0.7)
                    branch_current: lp.Point = lp.Point(x=int(current.x), y=int(current.y))
                    #
                    for bstep in range(branch_len):
                        #
                        blen: float = random.uniform(3.0, max(4.0, c_data.size / 30.0))
                        bdx: int = int(round(blen * cos(branch_angle) + random.uniform(-1.0, 1.0)))
                        bdy: int = int(round(blen * sin(branch_angle) + random.uniform(-1.0, 1.0)))
                        bpt: lp.Point = lp.Point(x=int(branch_current.x + bdx), y=int(branch_current.y + bdy))
                        #
                        if not (bpt in c_poly):
                            #
                            break
                        #
                        bt: float = float(bstep) / float(max(1, branch_len - 1))
                        bprof: float = max(0.0, 1.0 - 4.0 * (bt - 0.5) ** 2)
                        belev: float = branch_peak * bprof + random.uniform(-branch_peak * 0.1, branch_peak * 0.1)
                        belev = max(-500.0, belev)
                        #
                        bnode: lp.Point = lp.Point(x=int(bpt.x), y=int(bpt.y))
                        bnode.param3 = float(belev)
                        bnode.param2 = 0.0
                        bnode.param4 = 0.0
                        bnode.param1 = 0.0
                        bnode.param5 = 0.0
                        #
                        mountains_lp.append(bnode)
                        all_pts_hm.append(bnode)
                        #
                        branch_current = bpt
                        branch_angle += random.uniform(-pi / 8.0, pi / 8.0)

        #
        # 3) Hills and plains: sample interior points and take local mountain influence into account.
        #
        # approximate area ~ pi * r^2 (r ~ c_data.size), choose density parameter
        #
        area_approx: float = pi * (max(1.0, c_data.size) ** 2)
        #
        points_density_area: float = 30.0 * 30.0
        #
        n_interior_points: int = int(min(max(300, area_approx / points_density_area), 4000))
        #
        influence_sigma: float = max(4.0, c_data.size / 4.0)

        #
        for _ in range(n_interior_points):

            #
            p: lp.Point = c_poly.generate_random_point_uniformly()
            #
            if p is None:
                #
                continue

            #
            # base plain elevation
            #
            base_plain: float = random.uniform(0.0, 220.0) * size_scale

            #
            # mountain influence from nearby mountain points (if any)
            #
            near_mountains: list[lp.Point] = mountains_lp.get_all_points_in_circle(point=p, radius=influence_sigma)
            #
            extra: float = 0.0
            #
            if near_mountains:
                #
                # pick the maximum nearby mountain height and apply gaussian falloff
                #
                best_h: float = max([m.param3 for m in near_mountains])
                #
                # distance to the closest mountain (approx)
                #
                dclosest: float = min([p.calculate_distance(m) for m in near_mountains])
                #
                falloff: float = float(np.exp(- (dclosest ** 2) / (2.0 * (influence_sigma ** 2))))
                #
                extra = best_h * falloff * random.uniform(0.25, 0.9)

            #
            final_h: float = max(-400.0, base_plain + extra + random.uniform(-30.0, 30.0))

            #
            hpt: lp.Point = lp.Point(x=int(p.x), y=int(p.y))
            hpt.param2 = 0.0
            hpt.param3 = float(final_h)
            hpt.param4 = 0.0
            hpt.param1 = 0.0
            hpt.param5 = 0.0

            #
            all_pts_hm.append(hpt)

        #
        # 4) Rivers: start some rivers at high mountain nodes and route to nearest border.
        #
        # collect candidate mountain nodes
        #
        # (use get_all_points to avoid iterating over subclusters directly)
        #
        m_all = mountains_lp.get_all_points()
        #
        if m_all.size == 0:
            #
            continue

        #
        # create a small list of mountain points (as Point objects) from the structures we appended earlier
        #
        mountain_points_list: list[lp.Point] = [lp.Point(from_data=m_all[idx]) for idx in range(m_all.shape[0])]

        #
        # threshold to start rivers: top fraction of mountains
        #
        if mountain_points_list:
            #
            heights_list = [mp.param3 for mp in mountain_points_list]
            hsorted = sorted(heights_list, reverse=True)
            thresh = hsorted[max(0, int(len(hsorted) * 0.22))] if len(hsorted) > 0 else 0.0
        else:
            thresh = 9999.0

        #
        existing_river_nodes: list[tuple[int, int]] = []

        #
        for mp in mountain_points_list:

            #
            if mp.param3 < thresh:
                #
                continue

            #
            # probabilistic start (not every high mountain becomes river head)
            #
            if random.random() > 0.50:
                #
                continue

            #
            # find nearest border vertex as target (cheap linear search over border)
            #
            best_b: lp.Point = c_poly.boundary[0]
            best_d: float = mp.calculate_distance(best_b)
            for bi in range(1, len(c_poly.boundary)):
                bpt = c_poly.boundary[bi]
                d = mp.calculate_distance(bpt)
                if d < best_d:
                    best_d = d
                    best_b = bpt

            #
            # route the river by small steps biased toward the border target
            #
            current: lp.Point = lp.Point(x=int(mp.x), y=int(mp.y))
            river_size_val: float = max(1.0, min(6.0, mp.param3 / 800.0))
            #
            max_steps = int(max(100, best_d * 1.5))
            prev_node = current
            #
            for step in range(max_steps):

                #
                # direction vector to target
                #
                vx = float(best_b.x - current.x)
                vy = float(best_b.y - current.y)
                norm = sqrt(vx * vx + vy * vy)
                if norm == 0.0:
                    #
                    break
                #
                # step toward target with some lateral noise
                #
                sx = (vx / norm) * random.uniform(1.0, 4.0) + random.uniform(-1.5, 1.5)
                sy = (vy / norm) * random.uniform(1.0, 4.0) + random.uniform(-1.5, 1.5)

                nx = int(round(current.x + sx))
                ny = int(round(current.y + sy))

                #
                nxt = lp.Point(x=nx, y=ny)
                #
                # if out of polygon, project to nearest boundary location / stop
                #
                if not (nxt in c_poly):
                    #
                    # if close to boundary, snap to boundary and finish
                    #
                    if nxt.calculate_distance(best_b) <= 1.0:
                        #
                        nxt = lp.Point(x=int(best_b.x), y=int(best_b.y))
                        #
                        # append final segment and finish
                        #
                        rivers_segments.append((prev_node, nxt))
                        #
                        # add river nodes along final path
                        #
                        rn = lp.Point(x=int(nxt.x), y=int(nxt.y))
                        rn.param4 = float(river_size_val)
                        rn.param3 = float(min(rn.param3, 0.0))
                        all_pts_hm.append(rn)
                        existing_river_nodes.append((int(rn.x), int(rn.y)))
                        break
                    #
                    else:
                        #
                        break

                #
                # join existing river if close enough
                #
                close_join = False
                for (exx, exy) in existing_river_nodes[-200:]:
                    #
                    if (nx - exx) ** 2 + (ny - exy) ** 2 <= 9:
                        #
                        close_join = True
                        break

                if close_join:
                    #
                    rivers_segments.append((prev_node, nxt))
                    break

                #
                # register the river node
                #
                rn = lp.Point(x=int(nxt.x), y=int(nxt.y))
                rn.param4 = float(river_size_val)
                rn.param3 = float(max(-800.0, rn.param3))  # keep some elevation, rivers dig later
                rn.param2 = 0.0
                rn.param1 = 0.0
                rn.param5 = 0.0

                all_pts_hm.append(rn)
                rivers_segments.append((prev_node, rn))
                existing_river_nodes.append((int(rn.x), int(rn.y)))

                #
                prev_node = rn
                current = nxt

                #
                if current.calculate_distance(best_b) <= 3.0:
                    #
                    break

    #
    return all_pts_hm, rivers_segments



#
def create_heightmap_from_all_pts_heightmap(tx: int, ty: int, continent_polygons: list[lp.Polygon], all_pts_hm: lp.LargePointsAreas, rivers_segments: list[tuple[lp.Point, lp.Point]], water_base_height: int = -1000) -> NDArray[np.float32]:

    #
    hm: NDArray[np.float32] = np.ones( shape=(tx, ty), dtype=np.float32 ) * float(water_base_height)

    #
    # Get all points as a numpy array: columns -> x, y, param1,param2,param3,param4,param5
    #
    pts_arr: NDArray[np.float32] = all_pts_hm.get_all_points()

    #
    if pts_arr.size == 0:
        #
        return hm

    #
    xs: NDArray[np.int32] = pts_arr[:, 0].astype(np.int32)
    ys: NDArray[np.int32] = pts_arr[:, 1].astype(np.int32)
    #
    params2: NDArray[np.float32] = pts_arr[:, 3]  # border flag
    elevations: NDArray[np.float32] = pts_arr[:, 4]
    river_sizes: NDArray[np.float32] = pts_arr[:, 5]

    #
    # Useful normalization numbers
    #
    max_h = float(np.max(elevations))
    min_h = float(np.min(elevations))
    span_h = max(1e-6, max_h - min_h)

    #
    # Kernel painting parameters: mountains => sharper (small sigma), plains => broader.
    #
    min_sigma: float = 1.5
    max_sigma: float = 12.0

    #
    # Paint each point using a localized Gaussian kernel (vectorized in a small block)
    #
    npts = elevations.shape[0]
    for i in range(npts):

        #
        px = int(xs[i])
        py = int(ys[i])
        #
        if not (0 <= px < tx and 0 <= py < ty):
            #
            continue

        h = float(elevations[i])

        #
        # normalized elevation in [0,1]
        #
        if span_h <= 0.0:
            norm = 0.5
        else:
            norm = (h - min_h) / span_h

        #
        # mountains (high h) -> smaller sigma -> sharper peaks
        #
        sigma = float(max_sigma - norm * (max_sigma - min_sigma))
        sigma = max(0.8, sigma)

        #
        # bounding box for kernel (3 sigma)
        #
        radius = int(max(1, int(3.0 * sigma)))
        x0 = max(0, px - radius)
        x1 = min(tx, px + radius + 1)
        y0 = max(0, py - radius)
        y1 = min(ty, py + radius + 1)

        #
        # create local grid and compute gaussian
        #
        xs_block = np.arange(x0, x1, dtype=np.float32)
        ys_block = np.arange(y0, y1, dtype=np.float32)
        xv, yv = np.meshgrid(xs_block, ys_block, indexing='ij')  # shape (x_len, y_len)
        dist2 = (xv - float(px)) ** 2 + (yv - float(py)) ** 2
        kernel = np.exp(-dist2 / (2.0 * (sigma ** 2)))
        #
        # add contribution (mountains are positive we add to base ocean)
        #
        try:
            hm[x0:x1, y0:y1] += (h * kernel).astype(np.float32)
        except Exception:
            # Safety: in case of any odd shape mismatch, skip this point
            continue

    #
    # Rivers: carve channels by subtracting a small depth along river paths
    #
    for seg in rivers_segments:

        a: lp.Point = seg[0]
        b: lp.Point = seg[1]

        #
        # sample along segment
        #
        steps = max(1, int(round(a.calculate_distance(b) * 2.0)))
        for t_idx in range(steps + 1):

            t = float(t_idx) / float(max(1, steps))
            rx = int(round(a.x * (1.0 - t) + b.x * t))
            ry = int(round(a.y * (1.0 - t) + b.y * t))

            if not (0 <= rx < tx and 0 <= ry < ty):
                continue

            #
            # erosion depth and width depend on local river size if available (search nearest point)
            #
            # small heuristic: deeper near midpoints, shallow at ends
            #
            depth_center = 30.0
            width_center = 2

            #
            # carve a small rectangular neighborhood with gaussian-like falloff
            #
            for dx in range(-width_center*2, width_center*2 + 1):
                for dy in range(-width_center*2, width_center*2 + 1):
                    cx = rx + dx
                    cy = ry + dy
                    if 0 <= cx < tx and 0 <= cy < ty:
                        dist_sq = float(dx*dx + dy*dy)
                        fall = np.exp(-dist_sq / (2.0 * (max(0.5, float(width_center)) ** 2)))
                        hm[cx, cy] -= float(depth_center) * fall * 0.6 * 100

    #
    # Ensure points outside polygons are sea level and plains/hills have minimum height
    #
    pbar = tqdm(total=tx * ty)
    #
    for x in range(tx):
        for y in range(ty):
            inside_any = False
            for poly in continent_polygons:
                if lp.Point(x=x, y=y) in poly:
                    inside_any = True
                    break
            if not inside_any:
                #
                # outside any continent polygon -> set to sea level
                #
                hm[x, y] = float(water_base_height)
            else:
                #
                # inside polygon -> enforce plains/hills minimum
                # (if below 0, push to random between 0 and 50)
                #
                if hm[x, y] < 0.0:
                    hm[x, y] = random.uniform(0.0, 50.0)
            #
            pbar.update()
    #
    pbar.close()


    #
    # Final safety clamp: avoid dropping below some lower bound for ocean
    #
    # (we keep ocean baseline water_base_height but cap any over-erosion)
    #
    hm = np.maximum(hm, float(water_base_height - 5000.0)).astype(np.float32)

    #
    return hm



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
    all_points_for_heightmap: lp.LargePointsAreas = lp.LargePointsAreas(sub_cluster_size=10)

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
            []
            # generate_continent_points(
            #     tx=tx,
            #     ty=ty,
            #     continent_polygon=continent_polygons[_i],
            #     continent_data=initial_continents[_i],
            #     nb_initial_points=nb_initial_points,
            #     initial_points_strength=initial_points_strength,
            #     nb_per_points_generated=nb_per_points_generated,
            #     continent_superficy_min=continent_superficy_min,
            #     continent_superficy_max=continent_superficy_max,
            #     dist_between_points=dist_between_points
            # )
        )

        for _i in range(n_continents)

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
    all_points_for_heightmap: lp.LargePointsAreas
    rivers_segments: list[tuple[lp.Point, lp.Point]]
    #
    all_points_for_heightmap, rivers_segments = create_heightmap_points(
        tx=tx,
        ty=ty,
        continents_datas=initial_continents,
        continent_polygons=continent_polygons
    )

    #
    ### Generate an heightmap texture for all the planet. ###
    #
    arr: NDArray[np.float32] = create_heightmap_from_all_pts_heightmap(
        tx=tx,
        ty=ty,
        continent_polygons=continent_polygons,
        all_pts_hm=all_points_for_heightmap,
        rivers_segments=rivers_segments
    )

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

