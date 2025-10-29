from pathlib import Path

import h5py
import numpy as np
import unyt as u
from yt_experiments.octree.converter import OctTree

import yt


def load_dyablo(
    path: Path | str,
    field_mapping: dict | None = None,
    length_unit=None,
    mass_unit=None,
    time_unit=None,
    gamma0: float = 5 / 3,
    mean_molecular_weight: float = 1.2,
) -> yt.data_objects.static_output.Dataset:
    """Load a Dyablo HDF5 output file into a yt Dataset.

    Parameters
    ----------
    path : Path | str
        Path to the Dyablo HDF5 (field only) output file.
    field_mapping : dict | None, optional
        A mapping from Dyablo field names (e.g., 'e_tot') to a tuple of form
        (yt_field_name, units), by default None.
    length_unit : unyt.Unit | None, optional
        The length unit to use for the dataset, by default None.
    mass_unit : unyt.Unit | None, optional
        The mass unit to use for the dataset, by default None.
    time_unit : unyt.Unit | None, optional
        The time unit to use for the dataset, by default None.
    gamma0 : float, optional
        The adiabatic index to use for deriving pressure and temperature, by default 5/3.
    mean_molecular_weight : float, optional
        The mean molecular weight to use for deriving temperature, by default 1.2.

    Returns
    -------
    yt.data_objects.static_output.Dataset
        The loaded yt Dataset.
    """

    default_field_mapping = {
        "rho": ("density", "code_density"),
        "e_tot": ("e_tot", "code_density * code_velocity**2"),
        "rho_vx": ("specific_momentum_x", "code_density * code_velocity"),
        "rho_vy": ("specific_momentum_y", "code_density * code_velocity"),
        "rho_vz": ("specific_momentum_z", "code_density * code_velocity"),
    }

    default_field_mapping |= field_mapping or {}

    path = Path(path)
    with h5py.File(path, "r") as f:
        lower_left = f["coordinates"][:][f["connectivity"][:, 0]]
        upper_right = f["coordinates"][:][f["connectivity"][:, 6]]
        cell_sizes = upper_right - lower_left

        all_data = {
            k: f[k][:]
            for k in f
            if k not in ("coordinates", "connectivity", "scalar_data")
        }
        scalar_data = dict(f["scalar_data"].attrs)

    left_edge = lower_left.min(axis=0)
    right_edge = (lower_left + cell_sizes).max(axis=0)

    xc = lower_left + 0.5 * cell_sizes
    # Remap xc to [0, 1]
    dom_size = right_edge - left_edge
    xc = (xc - left_edge) / dom_size
    cell_sizes /= dom_size

    # Compute AMR level
    level = np.round(np.log2(1 / cell_sizes[:, 0])).astype(int)

    # Compute the order in which the cells appear in a depth-first octree traversal
    oct = OctTree.from_list(xc, level)
    ref_mask, leaf_order = oct.get_refmask()

    # Read in the data
    data = {}

    def reorder(dt):
        return np.where(leaf_order >= 0, dt[leaf_order], np.nan)[:, None].astype(
            np.float64
        )

    for k, v in all_data.items():
        k_dyablo, units = default_field_mapping.get(k, (k, "1"))
        # Make it 2D so that yt doesn't think those are particles
        data["gas", k_dyablo] = (reorder(v), units)

    ds = yt.load_octree(
        octree_mask=ref_mask,
        data=data,
        bbox=np.array([left_edge, right_edge]).T,
        num_zones=1,
        dataset_name=f"Dyablo/{path.name}",
        sim_time=scalar_data.get("time", 0.0),
        parameters=scalar_data,
        length_unit=length_unit,
        mass_unit=mass_unit,
        time_unit=time_unit,
    )

    # Derive velocities
    def gen_vel(component):
        def v(field, data):
            ftype = field.name[0]
            return (
                data[ftype, f"specific_momentum_{component}"] / data[ftype, "density"]
            )

        ds.add_field(
            ("gas", f"velocity_{component}"),
            sampling_type="local",
            function=v,
            units="code_velocity",
        )

    gen_vel("x")
    gen_vel("y")
    gen_vel("z")

    # Derive pressure
    def pressure(field, data):  # noqa: ARG001
        eK = (
            0.5
            * data["gas", "density"]
            * sum(data["gas", f"velocity_{k}"] ** 2 for k in ("x", "y", "z"))
        )
        eT = data["gas", "e_tot"] - eK
        return eT * (gamma0 - 1)

    ds.add_field(
        ("gas", "pressure"),
        sampling_type="local",
        function=pressure,
        units="code_pressure",
    )

    # Derive temperature
    def temperature(field, data):  # noqa: ARG001
        T = data["gas", "pressure"] / data["gas", "density"]
        return T * u.mp / u.kb * mean_molecular_weight

    ds.add_field(
        ("gas", "temperature"),
        sampling_type="local",
        function=temperature,
        units="K",
    )

    return ds
