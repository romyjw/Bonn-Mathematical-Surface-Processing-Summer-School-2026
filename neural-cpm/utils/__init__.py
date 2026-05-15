# Re-exports so notebooks can do e.g. `from utils import define_band_points`
from .define_band_points import define_band_points
from .Laplacian_matrix import Laplacian_matrix
from .RBF_update import phi, precompute_rbf_data, interpolate_from_precomputed
from .rot_update import (
    function_update,
    neural_extension,
    build_Global_dico,
    final_updated_function_value,
)
from .retrieve_neural_weights import retrieve_neural_weights
from .space_partition import space_partition
from .changing_local_basis import change_local_basis
from .draw import plot_3d_point_cloud
