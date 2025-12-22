import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, io

def read_cell_vector_fields_from_xdmf_h5(fiber_xdmf, comm):
    """
    Read meshio-written cell-data vectors (shape [num_cells_global, 3]) from the .h5 referenced by the XDMF.
    """
    import re
    import h5py
    fiber_data = {}

    try:
        h5_path = fiber_xdmf.replace(".xdmf", ".h5")
        with open(fiber_xdmf, "r") as f:
            xdmf_txt = f.read()

        # match Attribute Name="fiber_<something>"
        fiber_names = re.findall(r'<Attribute Name="(fiber_\w+)"', xdmf_txt)
        if comm.rank == 0:
            print(f"Found fiber/basis attributes in {fiber_xdmf}: {fiber_names}")

        with h5py.File(h5_path, "r") as h5:
            for name in fiber_names:
                # find the DataItem that points to the dataset
                # example: tube_mesh_fibers.h5:/data0
                pattern = rf'<Attribute Name="{name}".*?>(.*?)</Attribute>'
                attr_match = re.search(pattern, xdmf_txt, re.DOTALL)
                if attr_match is None:
                    continue
                data_match = re.search(r'>([^<]+\.h5:[^<]+)</DataItem>', attr_match.group(1))
                if data_match is None:
                    continue
                dataset_path = data_match.group(1).split(":")[1]
                arr = np.array(h5[dataset_path])
                fiber_data[name] = arr
                if comm.rank == 0:
                    print(f"  Loaded {name}: {arr.shape}")
        return fiber_data
    
    except Exception as e:
        raise RuntimeError(f"Error reading fiber data from {fiber_xdmf}: {e}")


def read_mesh(path):
    """Read mesh from XDMF file."""
    try:
        if path.endswith(".msh"):
            # from dolfinx.io import gmshio
            domain, _ = io.gmsh.read_from_msh(path, MPI.COMM_WORLD, gdim=3)
            return domain
        
        comm = MPI.COMM_WORLD
        with io.XDMFFile(comm, path, "r") as xdmf:
            domain = xdmf.read_mesh(name="Grid", ghost_mode=mesh.GhostMode.shared_facet)
        return domain
    
    except Exception as e:
        raise RuntimeError(f"Error reading mesh from {path}: {e}")


def read_meshtags(path, domain):
    """Read meshtags from XDMF file."""
    try:
        if path.endswith(".msh"):
            from dolfinx.io import gmshio
            _, meshtags = gmshio.read_meshtags_from_msh(path, domain, MPI.COMM_WORLD)
            return meshtags
        
        comm = MPI.COMM_WORLD
        with io.XDMFFile(comm, path, "r") as xdmf:
            meshtags = xdmf.read_meshtags(domain, name="Grid")
        return meshtags
    
    except Exception as e:
        raise RuntimeError(f"Error reading meshtags from {path}: {e}")


def read_fiber_data(path, comm):
    """Read fiber data from XDMF/H5 file."""
    try:
        if path.endswith(".msh"):
            from dolfinx.io import gmshio
            fiber_data = gmshio.read_cell_vector_fields_from_msh(path, comm)
            return fiber_data
        
        fiber_data = read_cell_vector_fields_from_xdmf_h5(path, comm)
        return fiber_data
    
    except Exception as e:
        raise RuntimeError(f"Error reading fiber data from {path}: {e}")


def build_DG0_vector_functions(domain, path, comm):
    """
    Map global cell-data arrays -> DG0 vector Functions (one vector per cell).
    Includes ghost cells and scatter_forward() for parallel correctness.  (FIX 6)
    """
    try:
        comm = domain.comm
        rank = comm.rank
        dim = domain.geometry.dim

        DG0v = fem.functionspace(domain, ("DG", 0, (dim,)))
        cell_imap = domain.topology.index_map(domain.topology.dim)

        n_local = cell_imap.size_local
        n_ghost = cell_imap.num_ghosts

        # IMPORTANT: include ghosts for correct evaluation/assembly on shared facets
        local_cells = np.arange(n_local + n_ghost, dtype=np.int32)
        global_cells = cell_imap.local_to_global(local_cells)

        fiber_data = read_fiber_data(path, comm)

        fiber_functions = {}
        for full_name, global_arr in fiber_data.items():
            short = full_name.replace("fiber_", "")
            f = fem.Function(DG0v, name=full_name)

            # Subset the global array by the (owned+ghost) global cell ids
            local_arr = global_arr[global_cells]  # shape (n_local+n_ghost, 3)

            # DG0 vector dof layout is flat [x0,y0,z0, x1,y1,z1, ...]
            f.x.array[: 3 * (n_local + n_ghost)] = local_arr.astype(np.float64).reshape(-1)
            f.x.scatter_forward()

            fiber_functions[short] = f
            if rank == 0:
                print(f"Created DG0 vector field: '{short}'")
        return fiber_functions
    
    except Exception as e:
        raise RuntimeError(f"Error building DG0 vector functions from {path}: {e}")