
#
from random import randint, choice
#
from math import floor
#
from PIL import Image, ImageDraw


#
output_path: str = "tests_lab/"


#
room_t = tuple[int, int, int]
color_t = tuple[int, int, int]
vec2_t = tuple[int, int]
vec4_t = tuple[int, int, int, int]


#
def random_room(tx: int, ty: int, tz: int) -> room_t:
    #
    return (
        randint(0, tx-1),
        randint(0, ty-1),
        randint(0, tz-1)
    )


#
def sort_rooms(room1: room_t, room2: room_t) -> tuple[room_t, room_t]:
    #
    if room1 <= room2:
        #
        return (room1, room2)
    #
    return (room2, room1)


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
    room1, room2 = sort_rooms(room1=room1, room2=room2)

    #
    if room1 not in doors:
        #
        return False

    #
    return room2 in doors[room1]


#
def add_door(doors: dict[room_t, set[room_t]], room1: room_t, room2: room_t) -> None:

    #
    room1, room2 = sort_rooms(room1=room1, room2=room2)

    #
    if room1 not in doors:
        #
        doors[room1] = set()

    #
    doors[room1].add( room2 )


#
def init_clusters(tx: int, ty: int, tz: int, rooms_clusters: dict[room_t, int], rooms_of_cluster: dict[int, set[room_t]]) -> int:

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
                rooms_clusters[rroom] = n_clusters
                #
                rooms_of_cluster[n_clusters] = set([rroom])
                #
                n_clusters += 1

    #
    return n_clusters


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
def render_labyrinth_floor(z: int, tx: int, ty: int, tz: int, first_room: room_t, end_room: room_t, rooms_clusters: dict[room_t, int], doors: dict[room_t, set[room_t]], rooms_to_avoid: set[room_t]) -> None:

    #
    margins: int = 10
    #
    room_size: int = 20
    wall_thick: int = 1
    door_size: int = 6
    wall_with_door_size: int = min(room_size//2, max(0, (room_size - door_size) // 2))
    #
    tsize: int = room_size // 4
    half_tsize: int = tsize // 2
    #
    first_room_color: color_t = (255, 0, 0)
    end_room_color: color_t = (0, 0, 255)
    #
    not_same_cluster_color: color_t = (200, 200, 200)
    neutral_room_color: color_t = (250, 250, 250)

    #
    im_width: int = 2 * margins + tx * room_size
    im_height: int = 2 * margins + ty * room_size
    #
    background_color: color_t = (255, 255, 255)
    #
    triangle_color: color_t = (0, 0, 0)
    wall_color: color_t = (0, 0, 0)

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
            room_rect_coords: vec4_t = (ix+wall_thick, iy+wall_thick, ix+room_size-2*wall_thick, iy+room_size-2*wall_thick)
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
                triangle_coords: list[ vec2_t ] = [(ix + 2 * tsize, iy + 2 * tsize), (ix + 3 * tsize, iy + 2 * tsize), (ix + tsize + half_tsize, iy + tsize)]
                #
                draw.polygon(triangle_coords, fill=triangle_color)

            #
            ### Draw bottom door. ###
            #
            if check_door(doors=doors, room1=room1, room2=bottom_room):
                #
                ### Lower triangle. ###
                #
                triangle_coords: list[ vec2_t ] = [(ix + tsize, iy + tsize), (ix + 2 * tsize, iy + 2 * tsize), (ix + tsize + half_tsize, iy + 2 * tsize)]
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
def create_labyrinth(tx: int, ty: int, tz: int = 1, begin_at_center: bool = False, random_end: bool = False, avoid_cycles: bool = False, rooms_to_avoid: set[room_t] = set()) -> None:

    #
    doors: dict[room_t, set[room_t]] = {}
    #
    rooms_clusters: dict[room_t, int] = {}
    #
    rooms_of_cluster: dict[int, set[room_t]] = {}
    #
    n_clusters: int = init_clusters(tx=tx, ty=ty, tz=tz, rooms_clusters=rooms_clusters, rooms_of_cluster=rooms_of_cluster)
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
        room2: room_t = choice(options)

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
if __name__ == "__main__":

    #
    rooms_to_avoid: list[room_t] = []

    #
    create_labyrinth(
        tx=20,
        ty=20,
        tz=1,
        begin_at_center=False,
        random_end=False,
        avoid_cycles=True,
        rooms_to_avoid=set(rooms_to_avoid)
    )
