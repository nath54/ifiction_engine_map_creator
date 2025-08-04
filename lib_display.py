# type: ignore
#
### Import modules. ###
#
import random
#
from matplotlib import pyplot as plt
#
import lib_points as lp

#
### Render points. ###
#
def render_points_from_clusters(tx: int, ty: int, points: lp.PointCluster) -> None:
    #
    ### Unpack the x and y coordinates from the PointCluster data. ###
    #
    x_coords = points.data[:points.length, 0]
    y_coords = points.data[:points.length, 1]

    #
    ### Create a scatter plot. ###
    #
    plt.figure()
    plt.scatter(x_coords, y_coords)

    #
    ### Set the title and labels. ###
    #
    plt.title("Rendered Points")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")

    # #
    # ### Set the plot limits based on the translation values. ###
    # #
    # plt.xlim(tx - 10, tx + 10)
    # plt.ylim(ty - 10, ty + 10)

    #
    ### Display the plot. ###
    #
    # plt.grid(True)
    # plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


#
### Render points with colors. ###
#
def render_points_with_colors_from_clusters(tx: int, ty: int, point_clusters: list[lp.PointCluster], colors: list[str]) -> None:

    #
    ### Check if the number of clusters matches the number of colors. ###
    #
    if len(point_clusters) != len(colors):
        #
        raise ValueError("The number of PointCluster objects must match the number of colors.")

    #
    ### Create a new plot. ###
    #
    plt.figure()

    #
    ### Iterate through each cluster and its corresponding color. ###
    #
    for i, cluster in enumerate(point_clusters):
        #
        ### Skip empty clusters. ###
        #
        if cluster.length > 0:
            #
            x_coords = cluster.data[:cluster.length, 0]
            y_coords = cluster.data[:cluster.length, 1]

            #
            ### Create a scatter plot for the current cluster. ###
            #
            plt.scatter(x_coords, y_coords, color=colors[i], label=f"Cluster {i+1}")

    #
    ### Set the title and labels. ###
    #
    plt.title("Rendered Points with Colors")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")

    # #
    # ### Set the plot limits based on the translation values. ###
    # #
    # plt.xlim(tx - 10, tx + 10)
    # plt.ylim(ty - 10, ty + 10)

    #
    ### Add a legend and display the plot. ###
    #
    # plt.legend()
    # plt.grid(True)
    # plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


#
### Render points. ###
#
def render_points_from_points_areas(tx: int, ty: int, points: lp.LargePointsAreas) -> None:
    #
    ### Unpack the x and y coordinates from the PointCluster data. ###
    #
    x_coords, y_coords = points.get_separate_coordinates_for_all_points()

    #
    ### Create a scatter plot. ###
    #
    plt.figure()
    plt.scatter(x_coords, y_coords)

    #
    ### Set the title and labels. ###
    #
    plt.title("Rendered Points")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")

    # #
    # ### Set the plot limits based on the translation values. ###
    # #
    # plt.xlim(tx - 10, tx + 10)
    # plt.ylim(ty - 10, ty + 10)

    #
    ### Display the plot. ###
    #
    # plt.grid(True)
    # plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


#
### Render points with colors. ###
#
def render_points_with_colors_from_points_areas(tx: int, ty: int, point_clusters: list[lp.LargePointsAreas], colors: list[str]) -> None:

    #
    ### Check if the number of clusters matches the number of colors. ###
    #
    if len(point_clusters) != len(colors):
        #
        raise ValueError("The number of PointCluster objects must match the number of colors.")

    #
    ### Create a new plot. ###
    #
    plt.figure()

    #
    ### Iterate through each cluster and its corresponding color. ###
    #
    for i, cluster in enumerate(point_clusters):
        #
        ### Skip empty clusters. ###
        #
        if cluster.length > 0:
            #
            x_coords, y_coords, _, is_border_point = cluster.get_separate_coordinates_for_all_points()

            #
            x_coords_non_border: list[float] = []
            y_coords_non_border: list[float] = []
            #
            x_coords_border: list[float] = []
            y_coords_border: list[float] = []
            #
            for jj in range(len(x_coords)):
                #
                if is_border_point[jj] == 1:
                    #
                    x_coords_border.append(x_coords[jj])
                    y_coords_border.append(y_coords[jj])
                #
                else:
                    #
                    x_coords_non_border.append(x_coords[jj])
                    y_coords_non_border.append(y_coords[jj])

            #
            ### Create a scatter plot for the current cluster. ###
            #
            # plt.scatter(x_coords, y_coords, color=colors[i], label=f"Cluster {i+1}")
            plt.scatter(x_coords_non_border, y_coords_non_border, marker=".", color=colors[i], label=f"Cluster {i+1}")
            plt.scatter(x_coords_border, y_coords_border, marker="x", color=colors[i], label=f"Cluster {i+1}")

    #
    ### Set the title and labels. ###
    #
    plt.title("Rendered Points with Colors")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")

    # #
    # ### Set the plot limits based on the translation values. ###
    # #
    # plt.xlim(tx - 10, tx + 10)
    # plt.ylim(ty - 10, ty + 10)

    #
    ### Add a legend and display the plot. ###
    #
    # plt.legend()
    # plt.grid(True)
    # plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


#
###
#
def generate_random_color() -> str:

    letters: str = "0123456789abcdef"

    #
    color = "#"

    #
    for i in range(6):

        #
        color += random.choice(letters)

    #
    return color


#
###
#
def generate_random_colors(n: int) -> list[str]:

    #
    return [ generate_random_color() for _ in range(n) ]

