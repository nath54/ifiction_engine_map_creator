#
### Import Modules. ###
#
import random
#
from tqdm import tqdm
#
import lib_points as lp
import lib_display as ld


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
def terrain_generator(
        tx: int = 2048,
        ty: int = 2048,
        nb_continents: int = 4,
        continent_superficy_min: int = 100,
        continent_superficy_max: int = 1000,
        dist_between_points: int = 20,
        treshold_point_continent_distance: float = 60,
        border_margin: int = 300,
        radius_border_points: float = 40,
        dead_angle_min_border_points: float = 100,
        radius_between_border_points: float = 10,
        search_radius_factor: float = 0.2,
    ) -> None:

    #
    points: list[ lp.Point ] = generate_continent_points(
        nb_continents=nb_continents,
        tx=tx,
        ty=ty,
        continent_superficy_min=continent_superficy_min,
        continent_superficy_max=continent_superficy_max,
        dist_between_points=dist_between_points,
        border_margin=border_margin,
    )

    #
    continents_points: list[ lp.LargePointsAreas ] = create_cluster_of_points(
        points=points,
        nb_continents=nb_continents,
        treshold_point_continent_distance=treshold_point_continent_distance
    )

    #
    print(f"Continents created: {len(continents_points)}")

    #
    polygons: list[lp.Polygon] = []

    #
    for cp in continents_points:
        #
        cp.set_all_point_border(radius=radius_border_points, dead_angle_min=dead_angle_min_border_points, radius_between_border_points=radius_between_border_points)
        #
        polygons.append( cp.create_polygon_from_border(search_radius_factor=search_radius_factor) )

    #
    ld.render_points_with_colors_from_points_areas_with_polygons(tx=tx, ty=ty, point_clusters=continents_points, colors=ld.generate_random_colors(len(continents_points)), polygons=polygons)

    #
    # TODO: calculate "border" points.

    # TODO: Add mountains, rivers, lakes, lands to continents. (rivers must have branches, and etc...). Separate ocean from continents.

    # TODO: Add cities to continents.

    # TODO: Create countries from cities / favorise natural countries borders like mountains chains or rivers.

    # TODO: Create Buildings for each city.

    # TODO: Create people for each city / country.


#
if __name__ == "__main__":

    #
    terrain_generator()
