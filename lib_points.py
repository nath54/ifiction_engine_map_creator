#
### Import Modules. ###
#
from typing import Optional, Any, Callable
#
import numpy as np
from numpy.typing import NDArray


#
### Numpy data type for representing points. ###
#
point_type = np.int32  # Coordinates in a 2D grid, so integers for now.
#
distance_type = np.float32  # Distances between points.

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
    def __init__(self, x: int = 0, y: int = 0, param1: int = 0, from_data: Optional[NDArray[point_type]] = None) -> None:

        #
        self.data: NDArray[point_type]
        #
        if from_data is not None:
            #
            self.data = from_data
        #
        else:
            #
            self.data = np.array( [x, y, param1], dtype=point_type )


    #
    ### repr function. ###
    #
    def __repr__(self) -> str:

        #
        return f"Point(x={self.data[0]}, y={self.data[1]}, param1={self.data[2]})"


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
### Class to represent a cluster of points. ###
#
class PointCluster:


    #
    ### Init function. Constructor. ###
    #
    def __init__(self, init_size: int, capacity_increase_factor: float = 2.0) -> None:

        #
        self.data: NDArray[point_type] = np.zeros( (init_size, 3), dtype=point_type )

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
            return np.zeros( (3, ), dtype=point_type )

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
            new_data = np.zeros((new_capacity, 3), dtype=point_type)
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
    ### Function that calculate the minimum distance between a point and a cluster of points. ###
    #
    def distance_from_point(self, point: Point) -> float:

        #
        if self.length == 0:
            #
            return float("inf")

        #
        ### Use broadcasting to subtract the point's x and y coordinates from the cluster's data.   ###
        ### self.data[:self.length, :2] has shape (N, 2), and point.data[:2] has shape (2,).        ###
        ### NumPy handles the subtraction correctly without creating an intermediate array.         ###
        #
        squared_diffs = (self.data[:self.length, :2] - point.data[:2]) ** 2

        #
        ### Sum the squared differences for each point to get the squared Euclidean distances. ###
        #
        squared_distances = np.sum(squared_diffs, axis=1)

        #
        ### Return the square root of the minimum squared distance. ###
        #
        return np.sqrt(np.min(squared_distances))


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
        all_points: NDArray[point_type] = np.zeros( shape=(self.length, 2), dtype=point_type )

        #
        i: int = 0

        #
        for sbclstr in self.sub_clusters.values():

            #
            l: int = sbclstr.length

            #
            all_points[i:i+l, :] = sbclstr.data[:l, :2]

            #
            i += l

        #
        return all_points


    #
    ### Function to get separate coordinates list for all points. ###
    #
    def get_separate_coordinates_for_all_points(self) -> tuple[ list[int], list[int] ]:

        #
        all_points: NDArray[point_type] = self.get_all_points()

        #
        x_coords: list[int] = all_points[:, 0].tolist()  # type: ignore
        y_coords: list[int] = all_points[:, 1].tolist()  # type: ignore

        #
        return x_coords, y_coords
