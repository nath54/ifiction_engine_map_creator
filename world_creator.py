#
### Import Modules. ###
#
import random
#
from math import floor, ceil, pi, cos, sin, sqrt
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
        idx_to_insert: int = random.randint(0, n-2)

        #
        p1: lp.Point = continents_boundary_points[idx_to_insert]
        p2: lp.Point = continents_boundary_points[idx_to_insert+1]

        #
        pc: lp.Point = ( p1 + p2 ) / 2
        pd: float = p1.calculate_distance( p2 ) / sqrt(2)

        #
        dx: int = round( random.uniform(-pd, pd) )
        dy: int = round( random.uniform(-pd, pd) )

        #
        new_point: lp.Point = pc + lp.Point( x=dx, y=dy )

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

    """
    Generate continent polygons (first 3 TODOs):
        - create a raw cloud of points for continents
        - prepare random InitialContinentData list if none provided (avoid overlaps)
        - create continent polygons by calling create_continent_polygon

    Returns:
        (continent_polygons, initial_continents_used, all_points_container)
    """

    # -------------------------
    # 1) Generate raw point cloud
    # -------------------------

    #
    print("Terrain generator: generating raw continent point cloud...")
    #
    raw_points: list[lp.Point] = generate_continent_points(
        nb_continents=nb_continents,
        tx=tx,
        ty=ty,
        continent_superficy_min=max(1, continent_superficy_min // 1),
        continent_superficy_max=max(1, continent_superficy_max // 1),
        dist_between_points=dist_between_points,
        border_margin=border_margin
    )

    #
    ### Put raw points in a LargePointsAreas context (many helper functions expect that). ###
    #
    all_points: lp.LargePointsAreas = lp.LargePointsAreas()

    #
    for p in raw_points:

        #
        ### append uses named arg in other places in this repo -> keep same form. ###
        #
        all_points.append(value=p)

    # -------------------------
    # 2) Prepare initial_continents list (non-overlapping)
    # -------------------------

    #
    if not initial_continents:

        #
        print("No initial_continents provided: generating random initial continent descriptors...")
        #
        generated: list[InitialContinentData] = []

        #
        ### bounds for sizes (pixels). We keep continents reasonably sized relative to the map. ###
        #
        min_radius = max(12, int(radius_border_points * 0.5))
        max_radius = max(min_radius + 1, int(min(tx, ty) // 6))

        #
        max_attempts = 1000
        attempts = 0

        #
        while len(generated) < nb_continents and attempts < max_attempts:

            #
            attempts += 1

            #
            ### pick center guarded by border. ###
            #
            cx = random.randint(border_margin, tx - 1 - border_margin)
            cy = random.randint(border_margin, ty - 1 - border_margin)

            #
            ### pick size (radius) inside sensible bounds. ###
            #
            size = random.randint(min_radius, max_radius)

            #
            ### choose boundary-point counts heuristically from size. ###
            #
            nb_base = max(6, int(size / 3))               # number of base boundary points
            nb_int = max(0, int(size / 2))                # interior/interpolated boundary points
            nb_added = max(0, int(nb_base / 3))           # added noisy border points

            #
            ### decide type and shape. ###
            #
            typ = "shape" if random.random() < 0.6 else "random_walk"
            shape = random.choice(["circle", "ellipse", "square"]) if typ == "shape" else "random_walk"

            #
            candidate = InitialContinentData(
                type=typ,
                shape=shape,
                center_position_x=cx,
                center_position_y=cy,
                size=float(size),
                nb_base_boundary_points=nb_base,
                nb_int_boundary_points=nb_int,
                nb_added_boundary_points=nb_added
            )

            #
            ### Overlap check: require center separation >= sum of radii + safety margin. ###
            #
            ok = True
            #
            for other in generated:

                #
                dx = candidate.center_position_x - other.center_position_x
                dy = candidate.center_position_y - other.center_position_y
                d = sqrt(dx * dx + dy * dy)

                #
                ### safety margin uses declared sizes plus a tunable extra (use dead_angle_min_border_points ###
                ### & radius_border_points to be conservative). ###
                #
                safety = candidate.size + other.size + max(radius_border_points, dead_angle_min_border_points, 20)

                #
                if d < safety:

                    #
                    ok = False
                    break

            #
            if ok:

                #
                generated.append(candidate)

        #
        ### If we failed to place all continents without overlap within attempts, fill the remaining ###
        ### using a fallback layout (spread on a circle) — best-effort to avoid overlaps. ###
        #
        if len(generated) < nb_continents:

            #
            print( f"Warning: could not place all continents without overlap after {attempts} attempts."
                    " Falling back to ring placement for remaining continents." )

            #
            center_map_x = tx // 2
            center_map_y = ty // 2
            #
            ring_radius = max(min(tx, ty) // 4, max_radius * 3)

            #
            while len(generated) < nb_continents:

                #
                angle = random.uniform(0, 2 * pi)
                cx = int(center_map_x + ring_radius * cos(angle))
                cy = int(center_map_y + ring_radius * sin(angle))
                size = int(max(min_radius, max_radius * 0.6))
                nb_base = max(6, int(size / 3))
                nb_int = max(0, int(size / 2))
                nb_added = max(0, int(nb_base / 3))
                shape = random.choice(["circle", "ellipse", "square"])

                #
                generated.append(InitialContinentData(
                    type="shape",
                    shape=shape,
                    center_position_x=max(border_margin, min(tx - 1 - border_margin, cx)),
                    center_position_y=max(border_margin, min(ty - 1 - border_margin, cy)),
                    size=float(size),
                    nb_base_boundary_points=nb_base,
                    nb_int_boundary_points=nb_int,
                    nb_added_boundary_points=nb_added
                ))

        #
        initial_continents = generated

    # -------------------------
    # 3) Create continent polygons
    # -------------------------

    #
    print("Creating continent polygons from initial descriptors (calling create_continent_polygon)...")
    #
    continent_polygons: list[lp.Polygon] = []

    #
    for cid, ic in enumerate(initial_continents):

        #
        print(f" - continent {cid+1}/{len(initial_continents)}: center=({ic.center_position_x},{ic.center_position_y}) size={ic.size} type={ic.type} shape={ic.shape}")
        #
        poly = create_continent_polygon(continent_id=cid, initial_continent_data=ic, all_points=all_points)
        continent_polygons.append(poly)

    #
    print(f"Done: created {len(continent_polygons)} continent polygons.")

    #
    ld.render_only_polygons(tx=tx, ty=ty, polygons=continent_polygons)

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
