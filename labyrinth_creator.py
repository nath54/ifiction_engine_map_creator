#
from typing import Any
#
import json
#
from random import randint
from random import choice as rchoice
#
from math import floor
#
from PIL import Image, ImageDraw
#
from tqdm import tqdm


#
output_path: str = "tests_lab/"


#
room_t = tuple[int, int, int]
color_t = tuple[int, int, int]
vec2_t = tuple[int, int]
vec3_t = tuple[int, int, int]
vec4_t = tuple[int, int, int, int]


#
MANHATAN_DIRECTIONS: list[vec3_t] = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]


#
def render_labyrinth_floor(z: int, tx: int, ty: int, tz: int, first_room: room_t, end_room: room_t, rooms_clusters: dict[room_t, int], doors: dict[room_t, set[room_t]], rooms_to_avoid: set[room_t]) -> None:

    #
    margins: int = 10
    #
    room_size: int = 60
    wall_thick: int = 2
    door_size: int = 24
    wall_with_door_size: int = min(room_size//2, max(0, (room_size - door_size) // 2))
    #
    tsize: int = 12
    half_tsize: int = 6
    #
    first_room_color: color_t = (255, 0, 0)
    end_room_color: color_t = (0, 255, 0)
    #
    not_same_cluster_color: color_t = (50, 50, 50)
    neutral_room_color: color_t = (10, 3, 6)

    #
    im_width: int = 2 * margins + tx * room_size
    im_height: int = 2 * margins + ty * room_size
    #
    background_color: color_t = (0, 0, 0)
    #
    wall_color: color_t = (140, 50, 80)
    #
    triangle_color: color_t = (70, 60, 0)

    #
    image: Image.Image

    #
    try:
        image = Image.new('RGB', (im_width, im_height), background_color)
        print("Empty image created with a white background.")
    except Exception as e:
        print(f"Error creating image: {e}")
        return

    #
    draw: ImageDraw.ImageDraw = ImageDraw.Draw(image)

    #
    ### Draw walls. ###
    #
    for x in range(tx+1):
        #
        ix: int = margins + x * room_size
        #
        rect_coords: vec4_t = (ix-wall_thick, margins, ix+wall_thick, im_height - margins)
        #
        if x == 0:
            #
            rect_coords: vec4_t = (ix, margins, ix+wall_thick, im_height - margins)
        #
        elif x == tx:
            #
            rect_coords: vec4_t = (ix-wall_thick, margins, ix, im_height - margins)
        #
        draw.rectangle(rect_coords, fill=wall_color)

    #
    for y in range(ty+1):
        #
        iy: int = margins + y * room_size
        #
        rect_coords: vec4_t = (margins, iy-wall_thick, im_width - margins, iy+wall_thick)
        #
        if y == 0:
            #
            rect_coords: vec4_t = (margins, iy, im_width - margins, iy+wall_thick)
        #
        elif y == ty:
            #
            rect_coords: vec4_t = (margins, iy-wall_thick, im_width - margins, iy)
        #
        draw.rectangle(rect_coords, fill=wall_color)

    #

    #
    for x in range(tx):
        #
        for y in range(ty):

            #
            room1: room_t = (x, y, z)

            #
            if room1 in rooms_to_avoid:
                #
                continue

            #
            ix: int = margins + x * room_size
            iy: int = margins + y * room_size

            #
            t1x: int = ix + tsize
            t1y: int = iy + tsize
            #
            t2x: int = ix + 5 * half_tsize
            t2y: int = iy + tsize

            #
            west_room: room_t   = (room1[0] - 1 , room1[1]      , room1[2]      )
            east_room: room_t   = (room1[0] + 1 , room1[1]      , room1[2]      )
            south_room: room_t  = (room1[0]     , room1[1] - 1  , room1[2]      )
            north_room: room_t  = (room1[0]     , room1[1] + 1  , room1[2]      )
            bottom_room: room_t = (room1[0]     , room1[1]      , room1[2] - 1  )
            top_room: room_t    = (room1[0]     , room1[1]      , room1[2] + 1  )

            #
            ### Get room floor color. ###
            #
            room_color: color_t = neutral_room_color
            #
            if rooms_clusters[room1] != rooms_clusters[first_room]:
                #
                room_color = not_same_cluster_color
            #
            elif room1 == first_room:
                #
                room_color = first_room_color
            #
            elif room1 == end_room:
                #
                room_color = end_room_color

            #
            ### Draw room floor. ###
            #
            room_rect_coords: vec4_t = (ix+wall_thick, iy+wall_thick, ix+room_size-wall_thick, iy+room_size-wall_thick)
            #
            draw.rectangle(room_rect_coords, fill=room_color)

            #
            ### Draw west wall. ###
            #
            if check_door(doors=doors, room1=room1, room2=west_room):
                #
                ### Wall with door. ###
                #
                rect_door: vec4_t = (ix, iy + wall_with_door_size, ix+wall_thick, iy+room_size-wall_with_door_size)
                #
                draw.rectangle(rect_door, fill=room_color)

            #
            ### Draw east wall. ###
            #
            if check_door(doors=doors, room1=room1, room2=east_room):
                #
                ### Wall with door. ###
                #
                rect_door: vec4_t = (ix+room_size-wall_thick, iy + wall_with_door_size, ix+room_size, iy+room_size-wall_with_door_size)
                #
                draw.rectangle(rect_door, fill=room_color)

            #
            ### Draw north wall. ###
            #
            if check_door(doors=doors, room1=room1, room2=north_room):
                #
                ### Wall with door. ###
                #
                rect_door: vec4_t = (ix + wall_with_door_size, iy+room_size-wall_thick, ix+room_size-wall_with_door_size, iy+room_size)
                #
                draw.rectangle(rect_door, fill=room_color)

            #
            ### Draw south wall. ###
            #
            if check_door(doors=doors, room1=room1, room2=south_room):
                #
                ### Wall with door. ###
                #
                rect_door: vec4_t = (ix + wall_with_door_size, iy, ix+room_size-wall_with_door_size, iy+wall_thick)
                #
                draw.rectangle(rect_door, fill=room_color)

            #
            ### Draw top door. ###
            #
            if check_door(doors=doors, room1=room1, room2=top_room):
                #
                ### Upper triangle. ###
                #
                triangle_coords: list[ vec2_t ] = [(t2x, t2y + tsize), (t2x + tsize, t2y + tsize), (t2x + half_tsize, t2y)]
                #
                draw.polygon(triangle_coords, fill=triangle_color)

            #
            ### Draw bottom door. ###
            #
            if check_door(doors=doors, room1=room1, room2=bottom_room):
                #
                ### Lower triangle. ###
                #
                triangle_coords: list[ vec2_t ] = [(t1x, t1y), (t1x + tsize, t1y), (t1x + half_tsize, t1y + tsize)]
                #
                draw.polygon(triangle_coords, fill=triangle_color)

    #
    output_filename = f"{output_path}labyrinth_floor_{z}.png"
    #
    try:
        #
        image.save(output_filename)
        print(f"Image saved as '{output_filename}'.")
    #
    except Exception as e:
        #
        print(f"Error saving image: {e}")


#
def render_labyrinth(tx: int, ty: int, tz: int, first_room: room_t, end_room: room_t, rooms_clusters: dict[room_t, int], doors: dict[room_t, set[room_t]], rooms_to_avoid: set[room_t]) -> None:

    #
    for z in range(tz):

        #
        render_labyrinth_floor(z=z, tx=tx, ty=ty, tz=tz, first_room=first_room, end_room=end_room, rooms_clusters=rooms_clusters, doors=doors, rooms_to_avoid=rooms_to_avoid)


#
def random_room(tx: int, ty: int, tz: int) -> room_t:
    #
    return (
        randint(0, tx-1),
        randint(0, ty-1),
        randint(0, tz-1)
    )


#
def sort_things(thing1: Any, thing2: Any) -> tuple[Any, Any]:
    #
    if thing1 <= thing2:
        #
        return (thing1, thing2)
    #
    return (thing2, thing1)


#
def get_first_and_last_rooms(tx: int, ty: int, tz: int, begin_at_center: bool = False, random_end: bool = False) -> tuple[room_t, room_t]:

    #
    first_room: room_t = (0, 0, 0)
    #
    if begin_at_center:
        #
        first_room = (floor(tx/2), floor(ty/2), floor(tz/2))

    #
    end_room: room_t = (tx - 1, ty - 1, tz - 1)
    #
    if random_end:

        #
        end_room = random_room(tx=tx, ty=ty, tz=tz)
        #
        while end_room == first_room:
            #
            end_room = random_room(tx=tx, ty=ty, tz=tz)

    #
    return (first_room, end_room)


#
def get_non_connected_neighbour_rooms(tx: int, ty: int, tz: int, doors: dict[room_t, set[room_t]], rroom: room_t) -> list[room_t]:

    #
    dx_options: list[int] = [0]
    dy_options: list[int] = [0]
    dz_options: list[int] = [0]
    #
    if rroom[0] > 0: dx_options.append(-1)
    if rroom[0] < tx - 1: dx_options.append(1)
    if rroom[1] > 0: dy_options.append(-1)
    if rroom[1] < ty - 1: dy_options.append(1)
    if rroom[2] > 0: dz_options.append(-1)
    if rroom[2] < tz - 1: dz_options.append(1)

    #
    options: list[room_t] = []
    #
    for dx in dx_options:
        #
        for dy in dy_options:
            #
            for dz in dz_options:

                #
                if abs(dx) + abs(dy) + abs(dz) != 1:
                    #
                    continue

                #
                nroom: room_t = (rroom[0]+dx, rroom[1]+dy, rroom[2]+dz)

                #
                if check_door(doors=doors, room1=rroom, room2=nroom):
                    #
                    continue

                #
                options.append( nroom )

    #
    return options


#
def check_door(doors: dict[room_t, set[room_t]], room1: room_t, room2: room_t) -> bool:

    #
    room1, room2 = sort_things(thing1=room1, thing2=room2)

    #
    if room1 not in doors:
        #
        return False

    #
    return room2 in doors[room1]


#
def add_door(doors: dict[room_t, set[room_t]], room1: room_t, room2: room_t) -> None:

    #
    room1, room2 = sort_things(thing1=room1, thing2=room2)

    #
    if room1 not in doors:
        #
        doors[room1] = set()

    #
    doors[room1].add( room2 )


#
def init_clusters(tx: int, ty: int, tz: int, rooms_clusters: dict[room_t, int], rooms_of_cluster: dict[int, set[room_t]], rooms_to_avoid: set[room_t]) -> int:

    #
    n_clusters: int = 0
    #
    for x in range(tx):
        #
        for y in range(ty):
            #
            for z in range(tz):
                #
                rroom: room_t = (x, y, z)
                #
                if rroom in rooms_to_avoid:
                    #
                    continue
                #
                rooms_clusters[rroom] = n_clusters
                #
                rooms_of_cluster[n_clusters] = set([rroom])
                #
                n_clusters += 1

    #
    return n_clusters


#
def init_adjacent_clusters(tx: int, ty: int, tz: int, rooms_clusters: dict[room_t, int], rooms_to_avoid: set[room_t], adjacents_clusters: dict[int, dict[int, dict[room_t, set[room_t]]]]) -> int:

    #
    tot_edges: set[tuple[room_t, room_t]] = set()

    #
    for x in range(tx):
        #
        for y in range(ty):
            #
            for z in range(tz):

                #
                for dx, dy, dz in MANHATAN_DIRECTIONS:

                    #
                    room1: room_t = (x, y, z)
                    room2: room_t = (x+dx, y+dy, z+dz)

                    #
                    if room2[0] < 0 or room2[1] < 0 or room2[2] < 0 or room2[0] >= tx or room2[1] >= ty or room2[2] >= tz or room2 in rooms_to_avoid:
                        #
                        continue

                    #
                    cluster1: int = rooms_clusters[room1]
                    cluster2: int = rooms_clusters[room2]

                    #
                    add_adjacents_clusters(adjacents_clusters=adjacents_clusters, cluster1=cluster1, cluster2=cluster2, room_cluster1=room1, room_cluster2=room2)

                    #
                    edges: tuple[room_t, room_t] = sort_things(thing1=room1, thing2=room2)
                    #
                    if edges not in tot_edges:
                        #
                        tot_edges.add(edges)

    #
    return len(tot_edges)


#
def merge_clusters(rooms_clusters: dict[room_t, int], rooms_of_cluster: dict[int, set[room_t]], n_clusters: int, room1: room_t, room2: room_t) -> int:

    #
    final_cluster: int = min( rooms_clusters[room1], rooms_clusters[room2] )
    #
    loosing_cluster: int = max( rooms_clusters[room1], rooms_clusters[room2] )

    #
    if final_cluster == loosing_cluster:
        #
        return n_clusters

    #
    for rroom in rooms_of_cluster[loosing_cluster]:

        #
        rooms_clusters[rroom] = final_cluster
        #
        rooms_of_cluster[final_cluster].add(rroom)

    #
    del rooms_of_cluster[loosing_cluster]

    #
    return (n_clusters - 1)


#
def add_adjacents_clusters(adjacents_clusters: dict[int, dict[int, dict[room_t, set[room_t]]]], cluster1: int, cluster2: int, room_cluster1: room_t, room_cluster2: room_t) -> None:

    #
    ### Depth 0. ###
    #
    if cluster1 not in adjacents_clusters:
        #
        adjacents_clusters[cluster1] = {}
    #
    if cluster2 not in adjacents_clusters:
        #
        adjacents_clusters[cluster2] = {}

    #
    ### Depth 1. ###
    #
    if cluster2 not in adjacents_clusters[cluster1]:
        #
        adjacents_clusters[cluster1][cluster2] = {}
    #
    if cluster1 not in adjacents_clusters[cluster2]:
        #
        adjacents_clusters[cluster2][cluster1] = {}

    #
    ### Depth 2. ###
    #
    if room_cluster1 not in adjacents_clusters[cluster1][cluster2]:
        #
        adjacents_clusters[cluster1][cluster2][room_cluster1] = set()
    #
    if room_cluster2 not in adjacents_clusters[cluster2][cluster1]:
        #
        adjacents_clusters[cluster2][cluster1][room_cluster2] = set()

    #
    ### Depth 3. ###
    #
    adjacents_clusters[cluster1][cluster2][room_cluster1].add( room_cluster2 )
    #
    adjacents_clusters[cluster2][cluster1][room_cluster2].add( room_cluster1 )


#
def merge_adjacents_clusters(adjacents_clusters: dict[int, dict[int, dict[room_t, set[room_t]]]], final_cluster: int, loosing_cluster: int) -> None:
    """
    Merge two clusters in the adjacents_clusters data structure.
    All adjacencies of loosing_cluster are transferred to final_cluster.
    """

    #
    if final_cluster == loosing_cluster:
        #
        return

    # Get all clusters adjacent to the loosing cluster
    # Use list() to avoid modification during iteration
    adjacent_to_loosing: list[int] = list(adjacents_clusters[loosing_cluster].keys())

    for cluster3 in adjacent_to_loosing:
        #
        if cluster3 == final_cluster:
            #
            continue

        # Transfer all room-to-room adjacencies from loosing_cluster to final_cluster
        for room_cluster2 in list(adjacents_clusters[loosing_cluster][cluster3].keys()):
            #
            for room_cluster3 in list(adjacents_clusters[loosing_cluster][cluster3][room_cluster2]):
                #
                add_adjacents_clusters(
                    adjacents_clusters=adjacents_clusters,
                    cluster1=final_cluster,
                    cluster2=cluster3,
                    room_cluster1=room_cluster2,
                    room_cluster2=room_cluster3
                )

    # Now safely remove all references to loosing_cluster
    # First pass: collect all clusters that reference loosing_cluster
    clusters_to_clean: list[int] = []
    for cluster3 in adjacent_to_loosing:
        if cluster3 in adjacents_clusters:  # Check if still exists
            #
            clusters_to_clean.append(cluster3)

    # Second pass: remove references to loosing_cluster
    for cluster3 in clusters_to_clean:
        #
        if cluster3 in adjacents_clusters and loosing_cluster in adjacents_clusters[cluster3]:
            #
            del adjacents_clusters[cluster3][loosing_cluster]

    # Finally, remove the loosing_cluster entirely
    if loosing_cluster in adjacents_clusters:
        #
        del adjacents_clusters[loosing_cluster]


#
def find_random_clusters_to_merge(adjacents_clusters: dict[int, dict[int, dict[room_t, set[room_t]]]]) -> tuple[int, int, room_t, room_t]:

    #
    cluster1: int = rchoice(list(adjacents_clusters.keys()))
    cluster2: int = rchoice(list(adjacents_clusters[cluster1].keys()))
    #
    cluster1, cluster2 = sort_things(thing1=cluster1, thing2=cluster2)
    #
    room_cluster1: room_t = rchoice(list(adjacents_clusters[cluster1][cluster2].keys()))
    room_cluster2: room_t = rchoice(list(adjacents_clusters[cluster1][cluster2][room_cluster1]))

    #
    return (cluster1, cluster2, room_cluster1, room_cluster2)


#
def create_labyrinth_algo_1(tx: int, ty: int, tz: int = 1, begin_at_center: bool = False, random_end: bool = False, avoid_cycles: bool = False, rooms_to_avoid: set[room_t] = set()) -> None:

    #
    doors: dict[room_t, set[room_t]] = {}
    #
    rooms_clusters: dict[room_t, int] = {}
    #
    rooms_of_cluster: dict[int, set[room_t]] = {}
    #
    n_clusters: int = init_clusters(tx=tx, ty=ty, tz=tz, rooms_clusters=rooms_clusters, rooms_of_cluster=rooms_of_cluster, rooms_to_avoid=rooms_to_avoid)
    #
    ndoors: int = 0
    #
    tot_doors: int = tx * ty * tz

    #
    first_room: room_t
    end_room: room_t
    #
    first_room, end_room = get_first_and_last_rooms(tx=tx, ty=ty, tz=tz, begin_at_center=begin_at_center, random_end=random_end)

    #
    while rooms_clusters[first_room] != rooms_clusters[end_room]:

        #
        room1: room_t = random_room(tx=tx, ty=ty, tz=tz)

        #
        options: list[room_t] = get_non_connected_neighbour_rooms(tx=tx, ty=ty, tz=tz, doors=doors, rroom=room1)

        #
        if not options:
            #
            print("no options")
            #
            continue

        #
        room2: room_t = rchoice(options)

        #
        if room2 in rooms_to_avoid:
            #
            print("room to avoid")
            continue

        #
        if avoid_cycles and rooms_clusters[room1] == rooms_clusters[room2]:
            #
            print("same cluster")
            continue

        #
        add_door(doors=doors, room1=room1, room2=room2)
        #
        ndoors += 1
        #
        n_clusters = merge_clusters(rooms_clusters=rooms_clusters, rooms_of_cluster=rooms_of_cluster, n_clusters=n_clusters, room1=room1, room2=room2)
        #
        print(f"ndoors = {ndoors} / {tot_doors} | n_clusters = {n_clusters}")

    #
    render_labyrinth(tx=tx, ty=ty, tz=tz, first_room=first_room, end_room=end_room, rooms_clusters=rooms_clusters, doors=doors, rooms_to_avoid=rooms_to_avoid)

    #
    labyrinth_to_ifiction(tx=tx, ty=ty, tz=tz, doors=doors, first_room=first_room, end_room=end_room)


#
def create_labyrinth_algo_2(tx: int, ty: int, tz: int = 1, begin_at_center: bool = False, random_end: bool = False, avoid_cycles: bool = False, rooms_to_avoid: set[room_t] = set()) -> None:

    #
    doors: dict[room_t, set[room_t]] = {}
    #
    rooms_clusters: dict[room_t, int] = {}
    #
    rooms_of_cluster: dict[int, set[room_t]] = {}
    #
    adjacents_clusters: dict[int, dict[int, dict[room_t, set[room_t]]]] = {}
    #
    n_clusters: int = init_clusters(tx=tx, ty=ty, tz=tz, rooms_clusters=rooms_clusters, rooms_of_cluster=rooms_of_cluster, rooms_to_avoid=rooms_to_avoid)
    #
    _tot_doors = init_adjacent_clusters(tx=tx, ty=ty, tz=tz, rooms_clusters=rooms_clusters, rooms_to_avoid=rooms_to_avoid, adjacents_clusters=adjacents_clusters)
    #
    ndoors: int = 0
    #
    first_room: room_t
    end_room: room_t
    #
    first_room, end_room = get_first_and_last_rooms(tx=tx, ty=ty, tz=tz, begin_at_center=begin_at_center, random_end=random_end)

    #
    pbar = tqdm(total=n_clusters)

    #
    while n_clusters > 1:

        #
        room1: room_t
        room2: room_t

        #
        cluster1, cluster2, room1, room2 = find_random_clusters_to_merge(adjacents_clusters=adjacents_clusters)

        #
        add_door(doors=doors, room1=room1, room2=room2)
        #
        ndoors += 1
        #
        merge_adjacents_clusters(adjacents_clusters=adjacents_clusters, final_cluster=cluster1, loosing_cluster=cluster2)
        #
        n_clusters = merge_clusters(rooms_clusters=rooms_clusters, rooms_of_cluster=rooms_of_cluster, n_clusters=n_clusters, room1=room1, room2=room2)
        #
        pbar.update(n=1)
        #
        # print(f"ndoors = {ndoors} / {tot_doors} | n_clusters = {n_clusters}")

    #
    render_labyrinth(tx=tx, ty=ty, tz=tz, first_room=first_room, end_room=end_room, rooms_clusters=rooms_clusters, doors=doors, rooms_to_avoid=rooms_to_avoid)

    #
    labyrinth_to_ifiction(tx=tx, ty=ty, tz=tz, doors=doors, first_room=first_room, end_room=end_room)


#
def get_room_id(rroom: room_t) -> str:
    #
    return f"room_{rroom[0]}_{rroom[1]}_{rroom[2]}"


#
def get_door_id(room1: room_t, room2: room_t) -> str:
    #
    room1, room2 = sort_things(thing1=room1, thing2=room2)
    #
    return f"door_{room1[0]}_{room1[1]}_{room1[2]}_to_{room2[0]}_{room2[1]}_{room2[2]}"


#
def get_room_directions(room1: room_t, room2: room_t) -> tuple[str, str]:
    #
    dx: int = room1[0] - room2[0]
    dy: int = room1[1] - room2[1]
    dz: int = room1[2] - room2[2]
    #
    if dx == 1:
        #
        return "west", "east"
    #
    elif dx == -1:
        #
        return "east", "west"
    #
    elif dy == 1:
        #
        return "south", "north"
    #
    elif dy == -1:
        #
        return "north", "south"
    #
    elif dz == 1:
        #
        return "down", "up"
    #
    return "up", "down"


#
def labyrinth_to_ifiction(tx: int, ty: int, tz: int, doors: dict[room_t, set[room_t]], first_room: room_t, end_room: room_t) -> None:

    jsongame: dict[str, Any] = {
        "game_name": "Basic Test 1",
        "game_description": "Simple test",
        "game_author": "Nathan Cerisara (aka github.com/nath54)",
        "things": {
            "player1": {
                "type": "player",
                "id": "player1",
                "name": "Player",
                "room": "room1",
                "inventory": {},
                "missions": []
            },
        },
        "rooms": {
        },
        "variables": {},
        "players": ["player1"],
        "nb_turns": 0
    }

    #
    doors_to_append: set[tuple[room_t, room_t]] = set()

    #
    for x in range(tx):
        #
        for y in range(ty):
            #
            for z in range(tz):
                #
                rroom: room_t = (x, y, z)
                #
                room_id: str = get_room_id(rroom=rroom)
                #
                jsongame["rooms"][room_id] = {
                    "room_name": "room",
                    "description": "a simple empty room",
                    "accesses": [],
                    "things_inside": {}
                }
                #
                if rroom in doors:
                    #
                    for nrooms in doors[rroom]:
                        #
                        doors_to_append.add(
                            sort_things(thing1=rroom, thing2=nrooms)
                        )

    #
    for room1, room2 in doors_to_append:
        #
        door_id: str = get_door_id(room1=room1, room2=room2)
        #
        jsongame["things"][door_id] = {
            "type": "object",
            "id": door_id,
            "name": "door",
            "description": "a simple door",
            "attributes": ["openable"]
        }
        #
        room1_id: str = get_room_id(rroom=room1)
        room2_id: str = get_room_id(rroom=room2)
        #
        dir_to_room2, dir_to_room1 = get_room_directions(room1=room1, room2=room2)

        #
        jsongame["rooms"][room1_id]["things_inside"][door_id] = 1
        jsongame["rooms"][room2_id]["things_inside"][door_id] = 1
        #
        jsongame["rooms"][room1_id]["accesses"].append( {"thing_id": door_id, "direction": dir_to_room2, "links_to": room2_id} )
        jsongame["rooms"][room2_id]["accesses"].append( {"thing_id": door_id, "direction": dir_to_room1, "links_to": room1_id} )

    #
    ### Add player to first room. ###
    #
    first_room_id: str = get_room_id(rroom=first_room)
    #
    jsongame["things"]["player1"]["room"] = first_room_id
    jsongame["rooms"][first_room_id]["things_inside"]["player1"] = 1

    #
    ### Indicate the end room. ###
    #
    end_room_id: str = get_room_id(rroom=end_room)
    #
    jsongame["rooms"][end_room_id]["description"] = "This is the final round. You reached the end of the labyrinth."

    #
    with open("ifiction_save/game.json", "w", encoding="utf-8") as f:
        #
        json.dump(obj=jsongame, fp=f)


#
if __name__ == "__main__":

    #
    rooms_to_avoid: list[room_t] = []

    #
    create_labyrinth_algo_2(
        tx=20,
        ty=20,
        tz=5,
        begin_at_center=False,
        random_end=False,
        avoid_cycles=True,
        rooms_to_avoid=set(rooms_to_avoid)
    )
