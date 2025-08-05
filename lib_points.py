#
### Import Modules. ###
#
from typing import Optional, Any, Callable
#
from math import ceil
#
import numpy as np
from numpy.typing import NDArray
#
from tqdm import tqdm


#
### Clamp method. ###
#
def clamp(value: Any, minimum: Any, maximum: Any) -> Any:

    #
    if value < minimum:
        #
        return minimum

    #
    if value > maximum:
        #
        return maximum

    #
    return value


#
### Numpy data type for representing points. ###
#
point_type = np.int32  # Coordinates in a 2D grid, so integers for now.
#
distance_type = np.float32  # Distances between points.

#
###
#
NB_PTS_PARAMS: int = 2

#
### Default values for magic_methods_to_delegate. ###
#
default_value_magic_methods_to_delegate: list[str] = [
    "__add__", "__sub__", "__mul__", "__truediv__", "__floordiv__", "__mod__", "__pow__",
    "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",
    "__neg__", "__pos__", "__abs__",
    "__radd__", "__rsub__", "__rmul__", "__rtruediv__", "__rfloordiv__", "__rmod__", "__rpow__"
]


#
### delegate_magic_methods_to_numpy. ###
#
def delegate_magic_methods_to_numpy(magic_methods_to_delegate: list[str] = default_value_magic_methods_to_delegate) -> Any:
    """
    A decorator factory to delegate magic methods to a specified attribute.
    """

    #
    magic_methods_that_returns_obj_value: set[str] = {
        "__add__", "__sub__", "__mul__", "__truediv__", "__floordiv__", "__mod__", "__pow__",
        "__neg__", "__pos__", "__abs__",
        "__radd__", "__rsub__", "__rmul__", "__rtruediv__", "__rfloordiv__", "__rmod__", "__rpow__"
    }

    #
    magic_methods_that_returns_value_directly: set[str] = {
        "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__"
    }

    #
    ### . ###
    #
    def delegate_magic_methods(cls: Any) -> Any:
        """A decorator to delegate magic methods to the 'data' attribute."""

        #
        ### make_method_that_returns_obj_value ###
        #
        def make_method_that_returns_obj_value(name: str) -> Callable[..., Any]:

            #
            def wrapper(self: Any, other: Any) -> Any:

                #
                ### Handle cases where the other operand is a Point object. ###
                #
                if isinstance(other, cls):
                    #
                    result_data = getattr(self.data, name)(other.data)

                #
                ### Handle cases where the other operand is a simple number. ###
                #
                else:
                    #
                    result_data = getattr(self.data, name)(other)

                #
                return cls(from_data=result_data)

            #
            return wrapper

        #
        ### make_r_method_that_returns_obj_value ###
        #
        def make_r_method_that_returns_obj_value(name: str) -> Callable[..., Any]:

            #
            def wrapper(self: Any, other: Any) -> Any:

                #
                ### The numpy method for '__radd__' is simply '__add__', but the operands are swapped. ###
                #
                numpy_method_name = name.lstrip('__r')
                result_data = getattr(self.data, numpy_method_name)(other)

                #
                return cls(from_data=result_data)

            #
            return wrapper

        #
        ### make_method_that_returns_directly ###
        #
        def make_method_that_returns_directly(name: str) -> Callable[..., Any]:

            #
            def wrapper(self: Any, other: Any) -> bool:

                #
                if isinstance(other, cls):
                    #
                    result_data = getattr(self.data, name)(other.data).all()
                #
                else:
                    #
                    result_data = getattr(self.data, name)(other).all()

                #
                return result_data

            #
            return wrapper

        #
        ### Adding all the asked magic method to the class. ###
        #
        for name in magic_methods_to_delegate:

            #
            if name in magic_methods_that_returns_obj_value:

                #
                ### Differentiate between regular and reverse methods. ###
                #
                if name.startswith('__r'):
                    #
                    setattr(cls, name, make_r_method_that_returns_obj_value(name))
                #
                else:
                    #
                    setattr(cls, name, make_method_that_returns_obj_value(name))

            #
            elif name in magic_methods_that_returns_value_directly:

                #
                setattr(cls, name, make_method_that_returns_directly(name))

        #
        return cls

    #
    return delegate_magic_methods


#
### Class to represent a single point of a 2D grid (world grid). ###
#
@delegate_magic_methods_to_numpy()
class Point:


    #
    ### Init function. Constructor. ###
    #
    def __init__(self, x: int = 0, y: int = 0, params: list[int] = [0 for _ in range(NB_PTS_PARAMS)], from_data: Optional[NDArray[point_type]] = None) -> None:

        #
        self.data: NDArray[point_type]
        #
        if from_data is not None:
            #
            self.data = from_data
        #
        else:
            #
            self.data = np.array( [x, y] + params, dtype=point_type )


    #
    ### repr function. ###
    #
    def __repr__(self) -> str:

        #
        return f"Point(x={self.data[0]}, y={self.data[1]}, params={self.data[2:]})"


    #
    ### x getter. ###
    #
    @property
    def x(self) -> int:
        """Getter for the x-coordinate."""

        #
        return self.data[0]


    #
    ### x setter. ###
    #
    @x.setter
    def x(self, value: int) -> None:
        """Setter for the x-coordinate, with type validation."""

        #
        if not isinstance(value, int):  # type: ignore
            #
            raise TypeError("x-coordinate must be an integer.")

        #
        self.data[0] = value


    #
    ### y getter. ###
    #
    @property
    def y(self) -> int:
        """Getter for the y-coordinate."""

        #
        return self.data[1]


    #
    ### y setter. ###
    #
    @y.setter
    def y(self, value: int) -> None:
        """Setter for the y-coordinate, with type validation."""

        #
        if not isinstance(value, int):  # type: ignore
            #
            raise TypeError("y-coordinate must be an integer.")

        #
        self.data[1] = value


    #
    ### param1 getter. ###
    #
    @property
    def param1(self) -> int:
        """Getter for the param1 value."""

        #
        return self.data[2]


    #
    ### param1 setter. ###
    #
    @param1.setter
    def param1(self, value: int) -> None:
        """Setter for the param1 value, with type validation."""

        #
        if not isinstance(value, int) and not isinstance(value, np.int32):  # type: ignore
            #
            raise TypeError(f"param1 must be an integer. Value asked to set param1 from = `{value}` of type `{type(value)}`")

        #
        self.data[2] = value


    #
    ### param2 getter. ###
    #
    @property
    def param2(self) -> int:
        """Getter for the param2 value."""

        #
        return self.data[3]


    #
    ### param2 setter. ###
    #
    @param2.setter
    def param2(self, value: int) -> None:
        """Setter for the param2 value, with type validation."""

        #
        if not isinstance(value, int) and not isinstance(value, np.int32):  # type: ignore
            #
            raise TypeError(f"param2 must be an integer. Value asked to set param2 from = `{value}` of type `{type(value)}`")

        #
        self.data[3] = value


    #
    ### Calculate angle with another point. ###
    #
    def calculate_angle(self, p: "Point") -> float:

        #
        if self.x == p.x:

            #
            if self.y == p.y:  return float("inf")

            #
            if self.y < p.y:  return 180.0

            #
            return 0.0

        #
        if self.y == p.y:

            #
            if self.x < p.x:  return 90.0

            #
            return 270.0

        #
        dx: float
        dy: float

        #
        if self.x > p.x:

            #
            dx = self.x - p.x

            #
            if self.y > p.y:  # Quadrant 2

                #
                dy = self.y - p.y

                #
                return 90.0 - ( np.atan( dy / dx ) * 180.0 / np.pi )

            #
            else:  # Quandrant 1

                #
                dy = p.y - self.y

                #
                return 90.0 + ( np.atan( dy / dx ) * 180.0 / np.pi )

        #
        dx = p.x - self.x

        #
        if self.y > p.y:  # Quandrant 3

            #
            dy = self.y - p.y

            #
            return 270.0 - ( np.atan( dy / dx ) * 180.0 / np.pi )

        # Quadrant 4

        #
        dy = p.y - self.y

        #
        return 270.0 + ( np.atan( dy / dx ) * 180.0 / np.pi )


    #
    ### Calculate distance with another point. ###
    #
    def calculate_distance(self, p: "Point") -> float:

        #
        return np.sqrt( np.sum( ( self.data[:2] - p.data[:2] ) ** 2 ) )


#
### Class to represent a cluster of points. ###
#
class PointCluster:


    #
    ### Init function. Constructor. ###
    #
    def __init__(self, init_size: int, capacity_increase_factor: float = 2.0) -> None:

        #
        self.data: NDArray[point_type] = np.zeros( (init_size, 2 + NB_PTS_PARAMS), dtype=point_type )

        #
        self.length: int = 0
        self.capacity: int = init_size
        #
        self.capacity_increase_factor: float = capacity_increase_factor


    #
    ### repr function. ###
    #
    def __repr__(self) -> str:

        #
        desc: str = ""

        #
        if self.length > 0:

            #
            center: NDArray[point_type] = self.cluster_center()

            #
            desc += f", center={center}"

            #
            desc += f", first_point={self.data[0]}"

            #
            desc += f", last_point={self.data[self.length-1]}"

        #
        return f"PointCluster(length={self.length}{desc})"


    #
    ### __len__ function. ###
    #
    def __len__(self) -> int:

        #
        return self.length


    #
    ### cluster_center function. ###
    #
    def cluster_center(self) -> NDArray[point_type]:

        #
        if self.length == 0:
            #
            return np.zeros( (2 + NB_PTS_PARAMS, ), dtype=point_type )

        #
        return np.mean( self.data, axis=-1)


    #
    ### increase_data_check function. ###
    #
    def increase_data_check(self) -> None:

        #
        ### Check if the cluster is at capacity and resize if necessary. ###
        #
        if self.length >= self.capacity:

            #
            new_capacity = int(self.capacity * self.capacity_increase_factor)

            #
            ### Avoid infinite loops on factors <= 1. ###
            #
            if new_capacity <= self.capacity:
                #
                new_capacity = self.capacity + 1

            #
            ### Create a new, larger array and copy the old data. ###
            #
            new_data = np.zeros((new_capacity, 2 + NB_PTS_PARAMS), dtype=point_type)
            new_data[:self.length] = self.data
            self.data = new_data
            self.capacity = new_capacity


    #
    ### append function. ###
    #
    def append(self, value: object) -> None:

        #
        ### Type check. ###
        #
        if not isinstance(value, Point):
            #
            raise TypeError("Can only append a Point object")

        #
        ### Increment data capacity if needed. ###
        #
        self.increase_data_check()

        #
        ### Append the new point. ###
        #
        self.data[self.length] = value.data  # type: ignore
        self.length += 1


    #
    ### insert function. ###
    #
    def insert(self, value: object, index: int) -> None:

        #
        ### Type check. ###
        #
        if not isinstance(value, Point):
            #
            raise TypeError("Can only insert a Point object")

        #
        ### Validate the index. ###
        #
        if not 0 <= index <= self.length:
            #
            raise IndexError("Index out of bounds")

        #
        ### Increment data capacity if needed. ###
        #
        self.increase_data_check()

        #
        ### Shift elements to the right to make space for the new point. ###
        #
        if index < self.length:
            #
            self.data[index + 1 : self.length + 1] = self.data[index : self.length]

        #
        ### Insert the new point. ###
        #
        self.data[index] = value.data  # type: ignore
        self.length += 1


    #
    ### __getitem__ function. ###
    #
    def __getitem__(self, key: int | slice) -> 'Point | PointCluster':

        #
        if isinstance(key, int):
            #
            if not -self.length <= key < self.length:
                #
                raise IndexError("Index out of bounds")
            #
            return Point(from_data=self.data[key])

        #
        elif isinstance(key, slice):  # type: ignore

            #
            ### Return a new PointCluster for slices ###
            #
            new_cluster = PointCluster(init_size=0)
            new_cluster.data = self.data[:self.length][key]
            new_cluster.length = new_cluster.data.shape[0]
            new_cluster.capacity = new_cluster.length
            #
            return new_cluster

        #
        else:
            #
            raise TypeError("Index must be an integer or a slice")


    #
    ### __setitem__ function. ###
    #
    def __setitem__(self, key: int, value: Point) -> None:

        #
        if not isinstance(value, Point):
            #
            raise TypeError("Can only set a Point object")

        #
        if not -self.length <= key < self.length:
            #
            raise IndexError("Index out of bounds")

        #
        if key < 0:
            #
            key += self.length

        #
        self.data[key] = value.data


    #
    ### __delitem__ function. ###
    #
    def __delitem__(self, key: int) -> None:

        #
        if not -self.length <= key < self.length:
            #
            raise IndexError("Index out of bounds")

        #
        while key < 0:
            #
            key += self.length

        #
        self.data = np.delete(self.data, key, axis=0)
        self.length -= 1
        self.capacity -= 1  # Note: The capacity shrinks, which is a side effect of using np.delete


    #
    ### __iter__ function. ###
    #
    def __iter__(self) -> 'PointCluster':

        #
        self._current_index = 0
        return self


    #
    ### __next__ function. ###
    #
    def __next__(self) -> Point:

        #
        if self._current_index < self.length:

            #
            point = self.data[self._current_index]
            self._current_index += 1

            #
            return Point(from_data=point)

        #
        raise StopIteration


    #
    ### __contains__ function. ###
    #
    def __contains__(self, item: Point) -> bool:

        #
        if not isinstance(item, Point):
            #
            return False

        #
        ### Check if any row in the cluster data matches the item's data. ###
        #
        return (self.data[:self.length] == item.data).all(axis=1).any()


    #
    ### __add__ function. ###
    #
    def __add__(self, other: 'PointCluster') -> 'PointCluster':

        #
        if not isinstance(other, PointCluster):  # type: ignore
            #
            return NotImplemented

        #
        new_cluster = PointCluster(self.length + other.length)
        new_cluster.data[:self.length] = self.data[:self.length]
        new_cluster.data[self.length:] = other.data[:other.length]
        new_cluster.length = self.length + other.length

        #
        return new_cluster


    #
    ### __iadd__ function. ###
    #
    def __iadd__(self, other: 'PointCluster | Point') -> 'PointCluster':

        #
        if isinstance(other, Point):
            #
            self.append(other)

        #
        elif isinstance(other, PointCluster):

            #
            new_length = self.length + other.length

            #
            while new_length > self.capacity:
                #
                self.increase_data_check()

            #
            self.data[self.length:new_length] = other.data[:other.length]
            self.length = new_length

        #
        else:
            #
            return NotImplemented

        #
        return self


    #
    ### Function that calculate all the distances between a point and the cluster of points. ###
    #
    def squared_distances_from_point(self, point: Point) -> NDArray[distance_type]:

        #
        squared_diffs = (self.data[:self.length, :2] - point.data[:2]) ** 2

        #
        ### Sum the squared differences for each point to get the squared Euclidean distances. ###
        #
        squared_distances = np.sum(squared_diffs, axis=1)

        #
        ### Return the square root of the minimum squared distance. ###
        #
        return squared_distances


    #
    ### Function that calculate the minimum distance between a point and a cluster of points. ###
    #
    def distance_from_point(self, point: Point) -> float:

        #
        if self.length == 0:
            #
            return float("inf")

        ### Sum the squared differences for each point to get the squared Euclidean distances. ###
        #
        squared_distances = self.squared_distances_from_point( point = point )

        #
        ### Return the square root of the minimum squared distance. ###
        #
        return np.sqrt(np.min(squared_distances))


    #
    ### Function to return all the points that are inside the sphere of a certain center and radius. ###
    #
    def get_all_points_inside_sphere(self, center: Point, radius: float) -> list[Point]:

        #
        if self.length == 0:

            #
            return []

        #
        squared_distances: NDArray[distance_type] = self.squared_distances_from_point( point = center )

        #
        ### Calculate a mask for all the points that are < (radius ** 2). ###
        #
        squared_radius = radius ** 2
        #
        mask = squared_distances < squared_radius

        #
        ### Get the data of all of these points and return a list of points corresponding to those points. ###
        #
        points_inside_data = self.data[:self.length][mask]

        #
        ### Return thoses points. ###
        #
        return [Point(from_data=data) for data in points_inside_data]


#
### Class to represent a large area of points. ###
#
class LargePointsAreas:

    #
    ### Init function. Constructor. ###
    #
    def __init__(self, sub_cluster_size: int = 100) -> None:

        #
        self.sub_cluster_size: int = sub_cluster_size

        #
        self.sub_clusters: dict[str, PointCluster] = {}

        #
        self.length: int = 0


    #
    ### Point to key function. ###
    #
    def point_to_key(self, p: Point) -> str:

        #
        sx: int = int(p.x // self.sub_cluster_size)
        sy: int = int(p.y // self.sub_cluster_size)

        #
        return f"{sx}_{sy}"


    #
    ### Sub cluster key and neighbours function. ###
    #
    def sub_cluster_key_and_neighbours(self, p: Point) -> list[str]:

        #
        lst: list[str] = []

        #
        sx: int = int(p.x // self.sub_cluster_size)
        sy: int = int(p.y // self.sub_cluster_size)

        #
        for dx in [-1, 0, 1]:
            #
            for dy in [-1, 0, 1]:
                #
                lst.append( f"{sx+dx}_{sy+dy}" )

        #
        return lst


    #
    ### append function. ###
    #
    def append(self, value: object) -> None:

        #
        ### Type check. ###
        #
        if not isinstance(value, Point):
            #
            raise TypeError("Can only append a Point object")

        #
        ### Get subcluster key. ###
        #
        cluster_key: str = self.point_to_key( p = value)

        #
        ### Check for cluster existance. ###
        #
        if cluster_key not in self.sub_clusters:

            #
            ### Create sub cluster if not exists. ###
            #
            self.sub_clusters[ cluster_key ] = PointCluster( init_size = 8, capacity_increase_factor = 2 )

        #
        ### Add point to sub cluster. ###
        #
        self.sub_clusters[ cluster_key ].append( value=value )

        #
        ### Increase length. ###
        #
        self.length += 1


    #
    ### __contains__ function. ###
    #
    def __contains__(self, item: Point) -> bool:

        #
        if not isinstance(item, Point):
            #
            return False

        #
        ### Get subcluster key. ###
        #
        cluster_key: str = self.point_to_key( p = item )

        #
        ### Check for cluster existance. ###
        #
        if cluster_key not in self.sub_clusters:

            #
            return False

        #
        ### Check if any row in the cluster data matches the item's data. ###
        #
        return self.sub_clusters[ cluster_key ].__contains__( item = item )


    #
    ### __iter__ function. ###
    #
    def __iter__(self) -> 'LargePointsAreas':

        #
        self._current_sub_clusters_keys: list[str] = list( self.sub_clusters )
        self._current_index = 0
        self._current_nb_clusters_keys: int = len(self._current_sub_clusters_keys)

        #
        if self._current_nb_clusters_keys > 0:

            #
            self.sub_clusters[ self._current_sub_clusters_keys[self._current_index] ].__iter__()

        #
        return self


    #
    ### __next__ function. ###
    #
    def __next__(self) -> Point:

        #
        if self._current_index < self._current_nb_clusters_keys:

            #
            try:

                #
                return self.sub_clusters[ self._current_sub_clusters_keys[self._current_index] ].__next__()

            #
            except StopIteration:

                #
                self._current_index += 1

                #
                if self._current_index < self._current_nb_clusters_keys:

                    #
                    self.sub_clusters[ self._current_sub_clusters_keys[self._current_index] ].__iter__()

                    #
                    return self.__next__()

        #
        raise StopIteration


    #
    ### __add__ function. ###
    #
    def __add__(self, other: 'LargePointsAreas') -> 'LargePointsAreas':

        #
        if not isinstance(other, LargePointsAreas):  # type: ignore
            #
            return NotImplemented

        #
        new_sub_cluster_size: int = self.sub_cluster_size

        #
        new_points_areas: LargePointsAreas = LargePointsAreas( sub_cluster_size=new_sub_cluster_size )

        #
        if other.sub_cluster_size == new_sub_cluster_size:

            #
            sub_cluster_keys: list[str] = list( set( list( self.sub_clusters.keys() ) + list( other.sub_clusters.keys() ) ) )

            #
            ### Adding all sub clusters to the new cluster. ###
            #
            for key in sub_cluster_keys:

                #
                ss: list[ PointCluster ] = []

                #
                ### Getting the sub clusters. ###
                #
                if key in self.sub_clusters:
                    #
                    ss.append( self.sub_clusters[key] )
                #
                if key in other.sub_clusters:
                    #
                    ss.append( other.sub_clusters[key] )

                #
                ### Adding the sub clusters to the new points areas. ###
                #
                if len( ss ) == 2:
                    #
                    new_points_areas.sub_clusters[ key ] = ss[0].__add__( ss[1] )
                #
                else:
                    #
                    new_points_areas.sub_clusters[ key ] = ss[0]

        #
        else:

            #
            for key in self.sub_clusters.keys():

                #
                new_points_areas.sub_clusters[ key ] = self.sub_clusters[ key ]

            #
            for point in iter( other ):

                #
                new_points_areas.append( point )

        #
        new_points_areas.length = self.length + other.length

        #
        return new_points_areas


    #
    ### Function that calculate the minimum distance between a point and a cluster of points. ###
    #
    def distance_from_point(self, point: Point, max_sub_clusters_radius_search: int = 8) -> float:

        #
        ### . ###
        #
        if self.length == 0:
            #
            return float("inf")

        #
        ### . ###
        #
        keys_to_test: list[str] = self.sub_cluster_key_and_neighbours( p = point )

        #
        min_dist: float = float("inf")
        good: bool = False

        #
        ### . ###
        #
        for key in keys_to_test:

            #
            if key in self.sub_clusters:

                #
                dist: float = self.sub_clusters[key].distance_from_point( point=point )

                #
                if dist < min_dist:

                    #
                    min_dist = dist

                    #
                    good = True

        #
        if good:
            #
            return min_dist

        #
        ### . ###
        #
        sx: int = int(point.x // self.sub_cluster_size)
        sy: int = int(point.y // self.sub_cluster_size)

        #
        radius: int = 1

        #
        while not good and radius < max_sub_clusters_radius_search:

            #
            radius += 1

            #
            ### . ###
            #
            nkeys_to_test: set[str] = set()

            #
            for nx in range(sx-radius, sx+radius):

                #
                nkeys_to_test.add( f"{nx}_{sy-radius}" )
                nkeys_to_test.add( f"{nx}_{sy+radius}" )

            #
            for ny in range(sy-radius, sy+radius):

                #
                nkeys_to_test.add( f"{sx-radius}_{ny}" )
                nkeys_to_test.add( f"{sx+radius}_{ny}" )

            #
            ### . ###
            #
            for key in nkeys_to_test:

                #
                if key in self.sub_clusters:

                    #
                    dist: float = self.sub_clusters[key].distance_from_point( point=point )

                    #
                    if dist < min_dist:

                        #
                        min_dist = dist

                        #
                        good = True

        #
        ### . ###
        #
        return min_dist


    #
    ### Function to get all points. ###
    #
    def get_all_points(self) -> NDArray[point_type]:

        #
        all_points: NDArray[point_type] = np.zeros( shape=(self.length, 2 + NB_PTS_PARAMS), dtype=point_type )

        #
        i: int = 0

        #
        for sbclstr in self.sub_clusters.values():

            #
            l: int = sbclstr.length

            #
            all_points[i:i+l, :] = sbclstr.data[:l, :]

            #
            i += l

        #
        return all_points


    #
    ### Function to get separate coordinates list for all points. ###
    #
    def get_separate_coordinates_for_all_points(self) -> tuple[ list[int], ... ]:

        #
        all_points: NDArray[point_type] = self.get_all_points()

        #
        x_coords: list[int] = all_points[:, 0].tolist()  # type: ignore
        y_coords: list[int] = all_points[:, 1].tolist()  # type: ignore

        #
        res: list[ list[int] ] = [ x_coords, y_coords ]

        #
        for i in range(NB_PTS_PARAMS):

            #
            res.append(
                all_points[:, 2+i].tolist()  # type: ignore
            )

        #
        return tuple( res )


    #
    ### Function to get all the subclusters that intersect with a certain circle. ###
    #
    def get_subclusters_in_radius(self, point: Point, radius: float) -> list[PointCluster]:

        #
        result_clusters: list[PointCluster] = []

        #
        center_sx: int = int( point.x / self.sub_cluster_size )
        center_sy: int = int( point.y / self.sub_cluster_size )

        #
        max_grid_dist = ceil(radius / self.sub_cluster_size)

        #
        radius_squared: float = radius ** 2

        #
        for sx in range( center_sx - max_grid_dist, center_sx + max_grid_dist + 1 ):

            #
            for sy in range( center_sy - max_grid_dist, center_sy + max_grid_dist + 1 ):

                #
                current_key: str = f"{sx}_{sy}"

                #
                min_x: int = sx * self.sub_cluster_size
                max_x: int = min_x + self.sub_cluster_size
                min_y: int = sy * self.sub_cluster_size
                max_y: int = min_y + self.sub_cluster_size

                #
                closest_x: int = clamp(point.x, min_x, max_x)
                closest_y: int = clamp(point.y, min_y, max_y)

                #
                dist_sq: float = (closest_x - point.x) ** 2 + (closest_y - point.y) ** 2

                #
                if dist_sq <= radius_squared and current_key in self.sub_clusters:

                    #
                    result_clusters.append( self.sub_clusters[current_key] )

        #
        return result_clusters


    #
    ### Function to get all the points close to a point. ###
    #
    def get_all_points_in_circle(self, point: Point, radius: float) -> list[Point]:

        #
        res_points: list[Point] = []

        #
        subclusters: list[PointCluster] = self.get_subclusters_in_radius( point=point, radius=radius )

        #
        cluster: PointCluster
        #
        for cluster in subclusters:

            #
            res_points += cluster.get_all_points_inside_sphere( center=point, radius=radius )

        #
        return res_points


    #
    ### Set all point border. ###
    #
    def set_all_point_border(self, radius: float = 100, dead_angle_min: float = 160, radius_between_border_points: float = 30) -> None:

        #
        print("Finding all the border points...")
        #
        pbar = tqdm(total=self.length)

        #
        nb_border_pts: int = 0

        #
        border_points: list[Point] = []

        #
        cluster: PointCluster
        #
        for cluster in self.sub_clusters.values():

            #
            for i in range(cluster.length):

                #
                point: Point = Point( from_data = cluster.data[i] )

                #
                points_in_circle: list[Point] = self.get_all_points_in_circle( point=point, radius=radius )

                #
                bad_point: bool = False

                #
                angles: list[float] = []

                #
                for pp in points_in_circle:

                    #
                    if pp.data[3] and point.calculate_distance(pp) <= radius_between_border_points:

                        #
                        bad_point = True

                    #
                    if not (point.x == pp.x and point.y == pp.y):

                        #
                        angles.append( point.calculate_angle(pp) )

                #
                if bad_point:

                    #
                    continue

                #
                angles.sort()

                #
                max_dist_between_angles: float = 0

                #
                nb_angles: int = len(angles)
                #
                for j in range( nb_angles ):

                    #
                    angle_1: float = angles[j]
                    angle_2: float = angles[(j+1) % nb_angles]

                    #
                    if j == nb_angles - 1:

                        #
                        mini_dst = abs( (angle_2 + 360) - angle_1 )

                    else:
                        #
                        mini_dst: float = abs( angle_2 - angle_1 )

                    #
                    if mini_dst > max_dist_between_angles:

                        #
                        max_dist_between_angles = mini_dst

                #
                # print(f"DEBUG | point {i} | max_dist_between_angles = {max_dist_between_angles}")

                #
                if max_dist_between_angles >= dead_angle_min:

                    #
                    ### Point is a border point. ###
                    #
                    border_points.append( point )
                    #
                    nb_border_pts += 1
                    #
                    cluster.data[i, 3] = 1  # point.param2 is if a point is a border point or not

                #
                pbar.update(n=1)

        #
        print(f"Border points found : {nb_border_pts}")


    #
    ### Helper function for angle calculation. ###
    #
    def _calculate_ccw_angle(self, p_prev: Point, p_current: Point, p_candidate: Point) -> float:
        """Calculates the counter-clockwise angle of the turn at p_current."""
        #
        v_ref: Point = p_current - p_prev
        v_cand: Point = p_candidate - p_current
        #
        angle: float = np.arctan2(v_cand.y, v_cand.x) - np.arctan2(v_ref.y, v_ref.x)
        #
        # Normalize to [0, 2*pi]
        if angle < 0:
            angle += 2 * np.pi
        #
        return angle


    #
    ### Algorithm: Spatially-Accelerated Angular Greedy Algorithm ###
    #
    def create_polygon_from_border(self, search_radius_factor: float = 3.0) -> 'Polygon':
        """
        Creates a cyclic path (polygon) from points marked as border points.

        Args:
            search_radius_factor:   Multiplier for the grid's sub_cluster_size to determine
                                    the initial search radius for the next point.

        Returns:
            A Polygon object representing the ordered boundary.
        """

        #
        print("Gathering border points...")

        #
        ### Step 0: Get all border points. Using a dict for O(1) lookups and removal. ###
        #
        unvisited_border_points: dict[Point, bool] = {
            p: True for p in self if p.param2 == 1
        }
        #
        if not unvisited_border_points:
            #
            print("No border points found to create a polygon.")
            return Polygon(PointCluster(0), self)

        #
        ### Step 1: Find the starting point (lowest y, then lowest x). ###
        #
        p_start = min(unvisited_border_points.keys(), key=lambda p: (p.y, p.x))
        #
        print(f"Starting polygon construction from point: {p_start}")

        #
        ### Step 2: Initialize for the main loop. ###
        #
        result_polygon_cluster = PointCluster(init_size=len(unvisited_border_points))
        result_polygon_cluster.append(p_start)
        #
        del unvisited_border_points[p_start]
        #
        p_current = p_start
        #
        ### Create a virtual previous point to establish a horizontal reference vector. ###
        #
        p_prev = Point(p_start.x - 10, p_start.y)

        #
        ### Step 3: main loop. ###
        #
        print("Constructing polygon path...")
        #
        pbar = tqdm(total=len(unvisited_border_points))

        #
        while unvisited_border_points:

            #
            ### Step 3a: Get candidate points efficiently. ###
            #
            search_radius = self.sub_cluster_size * search_radius_factor
            candidate_points = []

            #
            ### Adaptive search radius. ###
            #
            while not candidate_points:

                #
                points_in_circle = self.get_all_points_in_circle(point=p_current, radius=search_radius)
                #
                candidate_points = [p for p in points_in_circle if p in unvisited_border_points]

                #
                if not candidate_points:

                    #
                    ### # Increase radius if no candidates found. ###
                    #
                    new_search_radius: float = search_radius * 1.5

                    #
                    if search_radius > 10 * self.sub_cluster_size or search_radius > new_search_radius:
                        #
                        # print("\nWarning: Search radius is very large. There might be a large gap in border points.")

                        #
                        ### If search fails, fall back to all remaining points. ###
                        #
                        candidate_points = list(unvisited_border_points.keys())
                        break

                    #
                    search_radius = new_search_radius

            #
            ### Step 3b: Select the best candidate (tightest "left turn"). ###
            #
            best_candidate = None
            min_angle = float('inf') # We are looking for the smallest positive angle

            #
            for cand in candidate_points:

                #
                angle = self._calculate_ccw_angle(p_prev, p_current, cand)

                #
                if angle < min_angle:
                    #
                    min_angle = angle
                    best_candidate = cand

            #
            if best_candidate is None:
                #
                print("\nError: Could not find a next point. Aborting polygon creation.")
                break # Should not happen if candidate_points is not empty

            #
            ### Step 3d: Advance. ###
            #
            p_next: Point = best_candidate  # type: ignore
            #
            result_polygon_cluster.append(p_next)  # type: ignore
            #
            del unvisited_border_points[p_next]
            #
            p_prev = p_current  # type: ignore
            p_current = p_next  # type: ignore
            #
            pbar.update(1)

            #
            ### Step 3c: Check for loop termination. ###
            #
            if p_current == p_start:
                #
                print("\nWarning: Closed loop before visiting all points. This may indicate separate islands of border points.")
                break

        #
        pbar.close()
        print(f"Polygon construction complete with {len(result_polygon_cluster)} vertices.")

        #
        return Polygon(result_polygon_cluster, self)


#
### Class to represent a Polygon, with efficient point-in-polygon testing. ###
#
class Polygon:

    #
    ### Init function. Constructor. ###
    #
    def __init__(self, boundary: PointCluster, grid_context: LargePointsAreas):

        #
        self.boundary: PointCluster = boundary
        self.grid_context: LargePointsAreas = grid_context
        self.edge_grid: dict[str, list[int]] = {}
        #
        self.min_x: int = 0
        self.max_x: int = 0
        self.min_y: int = 0
        self.max_y: int = 0

        #
        if len(self.boundary) > 0:
            #
            self._calculate_bounding_box()
            self._precompute_edge_grid()


    #
    ### repr function. ###
    #
    def __repr__(self) -> str:
        #
        return f"Polygon(vertices={len(self.boundary)}, grid_cells_with_edges={len(self.edge_grid)})"


    #
    ### Calculate and store the polygon's axis-aligned bounding box. ###
    #
    def _calculate_bounding_box(self) -> None:
        #
        all_x: NDArray[point_type] = self.boundary.data[:self.boundary.length, 0]
        all_y: NDArray[point_type] = self.boundary.data[:self.boundary.length, 1]
        #
        self.min_x, self.max_x = np.min(all_x).item(), np.max(all_x).item()
        self.min_y, self.max_y = np.min(all_y).item(), np.max(all_y).item()


    #
    ### Pre-computation step to map grid cells to polygon edges. ###
    #
    def _precompute_edge_grid(self) -> None:
        """
        Iterates through each edge of the polygon, finds all grid cells it
        crosses, and stores a reference to the edge in those cells.
        """

        #
        print("Pre-computing polygon edge grid for fast lookups...")
        #
        for i in tqdm(range(len(self.boundary))):

            #
            p1 = self.boundary[i]
            p2 = self.boundary[(i + 1) % len(self.boundary)] # Wrap around for the last edge

            #
            ### Use a line traversal algorithm to find all intersected grid cells. ###
            #
            crossed_cells = self._get_grid_cells_for_line(p1, p2)

            #
            for cell_key in crossed_cells:

                #
                if cell_key not in self.edge_grid:
                    #
                    self.edge_grid[cell_key] = []

                #
                self.edge_grid[cell_key].append(i)

        #
        print("Edge grid pre-computation complete.")


    #
    ### Digital Differential Analyzer (DDA) based algorithm to find crossed cells. ###
    #
    def _get_grid_cells_for_line(self, p1: Point, p2: Point) -> set[str]:

        #
        cells: set[str] = set()
        #
        # grid_size: int = self.grid_context.sub_cluster_size

        #
        x1: int = p1.x
        y1: int = p1.y
        x2: int = p2.x
        y2: int = p2.y
        #
        dx: int = x2 - x1
        dy: int = y2 - y1

        #
        steps: int = max(abs(dx), abs(dy))
        #
        if steps == 0:
            #
            key = self.grid_context.point_to_key(p1)
            #
            cells.add(key)
            #
            return cells

        #
        x_inc: float = dx / steps
        y_inc: float = dy / steps
        #
        x: float = float(x1)
        y: float = float(y1)

        #
        for _ in range(int(steps) + 1):

            #
            key = self.grid_context.point_to_key(Point(int(x), int(y)))
            #
            cells.add(key)
            #
            x += x_inc
            y += y_inc

        #
        return cells

    #
    ### Helper for the ray casting intersection test. ###
    #
    def _ray_intersects_segment(self, point: Point, p1: Point, p2: Point) -> bool:

        #
        ### Assumes ray is cast horizontally to the right (positive x). ###
        #

        #
        ### Ensure p1.y <= p2.y for consistency. ###
        #
        if p1.y > p2.y:
            #
            p1, p2 = p2, p1

        #
        ### Case 1: Point y is outside the edge's y-range. No intersection. ###
        #
        if point.y == p1.y or point.y > p2.y:
            #
            return False

        #
        ### Case 2: Point y is on the same level as the lower vertex. ###
        ### This is a special case to avoid double counting when a ray hits a vertex. ###
        ### We only count it if the ray is also to the left of the vertex. ###
        #
        if point.y == p2.y:
            return point.x <= p2.x

        #
        ### Case 3: Edge is horizontal. Cannot intersect a horizontal ray unless co-linear. ###
        #
        if p1.y == p2.y:
            return False

        #
        ### Case 4: General case. Use line equation to find intersection x. ###
        ### Check if the point's x is to the left of the intersection point. ###
        ### This avoids floating point issues by using cross-product. ###
        ### (p2.y - p1.y) * (point.x - p1.x) - (p2.x - p1.x) * (point.y - p1.y) > 0 ###
        #
        if (p2.x - p1.x) * (point.y - p1.y) < (point.x - p1.x) * (p2.y - p1.y):
            #
            return True

        #
        return False


    #
    ### Algorithm: Grid-Accelerated Ray Casting ###
    #
    def __contains__(self, point: Point) -> bool:
        """
        Checks if a point is inside the polygon using a highly efficient
        grid-accelerated ray casting algorithm.

        Usage: `my_point in my_polygon`
        """

        #
        if len(self.boundary) < 3:
            #
            return False

        #
        ### Step 1: Trivial Rejection with Bounding Box Test. ###
        #
        if not (self.min_x <= point.x <= self.max_x and self.min_y <= point.y <= self.max_y):
            #
            return False

        #
        ### Step 2: Grid-Accelerated Ray Casting. ###
        #
        intersections: int = 0

        #
        ### We only need to check edges that could possibly intersect our horizontal ray. ###
        #
        #
        checked_edges: set[int] = set() # To avoid double-testing an edge if it's in multiple cells
        #
        gx_start: int = int(point.x // self.grid_context.sub_cluster_size)
        gy: int = int(point.y // self.grid_context.sub_cluster_size)
        gx_end: int = int(self.max_x // self.grid_context.sub_cluster_size)

        #
        gx: int
        #
        for gx in range(gx_start, gx_end + 1):

            #
            cell_key: str = f"{gx}_{gy}"
            #
            if cell_key in self.edge_grid:

                #
                for edge_index in self.edge_grid[cell_key]:

                    #
                    if edge_index in checked_edges:
                        #
                        continue

                    #
                    checked_edges.add(edge_index)
                    #
                    p1 = self.boundary[edge_index]
                    p2 = self.boundary[(edge_index + 1) % len(self.boundary)]

                    #
                    if self._ray_intersects_segment(point, p1, p2):
                        #
                        intersections += 1

        #
        ### Step 3: Odd number of intersections means the point is inside. ###
        #
        return intersections % 2 == 1


