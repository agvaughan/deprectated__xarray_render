
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='true'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.75'


import jax
import jax.numpy as jnp
from jax.scipy.stats.norm import pdf as jpdf
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import dask
import dask.array as da

import xarray as xr

from jax import device_put


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def initialize_image(coords_min,coords_max,coord_size,chunk_size,verbose=False):
    if verbose:
        for key,val in locals().items():
            print(key,val)
    coords_vector = np.linspace(coords_min,coords_max,coord_size)
    dims_coords_dict  = {'x': coords_vector, 'y':coords_vector,'z':coords_vector}                     

    arr = da.array(
    
    xarr = xr.DataArray(dask.,
                       dims=dims_coords_dict.keys(),
                       coords=dims_coords_dict).astype(arr_dtype).chunk(chunks=(chunk_size,chunk_size,chunk_size))
    return xarr

@jax.jit
def jax_broadcast_pdf_fn(x_loc,y_loc,z_loc,xx,yy,zz):
    '''
    For points in (xx,yy,zz), calculate distance to target-point at (x_loc,y_loc,z_loc).
    Then apply against the jpdf function defined in outer scope - typically this is gaussian.norm.pdf.
    
    '''
    return jpdf(jnp.sqrt((xx-x_loc)**2 + (yy-y_loc)**2 + (zz-z_loc)**2))

@jax.jit
def jax_broadcast_reduce_pdf_fn(x_locs,y_locs,z_locs,jxx,jyy,jzz):

    For points in (xx,yy,zz), calculate distance to target-points(s!) at (x_loc,y_loc,z_loc).

    Then apply against the jpdf function defined in outer scope - typically this is gaussian.norm.pdf to get a gaussian emission.

    Then sum across all of these distances to get the sum of multiple emitters.

    return jnp.squeeze(jpdf(jnp.sqrt((jxx-x_locs)**2 + (jyy-y_locs)**2 + (jzz-z_locs)**2)).sum(axis=-1))



def find_points_in_subimage(img,xyz_locs, pad_fraction, points_x, points_y, points_z):
    """
    INPUTS
        - img : xarray image, with labeled dimensions x,y,z
        - pad_fraction: scalar padding 0..1 to allow nearby points
        - points_x,points_y,points_z : vectors of points that may or may not be in the image.

    OUTPUTS
        - indices of the points that are within the (padded) image boundaries.
    """

    # Padded bounds around the subimage, which is a dask chunk of the full
    first = lambda dim: img[dim][0].values
    last = lambda dim: img[dim][-1].values
    padding = lambda dim: (last(dim) - first(dim)) * pad_fraction

    xmin = first("x") - padding("x")
    xmax = last("x") + padding("x")

    ymin = first("y") - padding("y")
    ymax = last("y") + padding("y")

    zmin = first("z") - padding("z")
    zmax = last("z") + padding("z")

    # Molecules within the (padded bounds of subimage)
    inds_in_image = np.where(
        (xyz_locs[:, 0] >= xmin)
        & (xyz_locs[:, 0] <= xmax)
        & (xyz_locs[:, 1] >= ymin)
        & (xyz_locs[:, 1] <= ymax)
        & (xyz_locs[:, 2] >= zmin)
        & (xyz_locs[:, 2] <= zmax)
    )[0]

    return inds_in_image


def render_subimage(sub_img,xyz_locs):
    jxx,jyy,jzz  = (jnp.array(coord) for coord in xr.broadcast(sub_img["x"], sub_img["y"], sub_img["z"]))    
    
    
    img_mol_inds = find_points_in_subimage(
        img=sub_img,
        xyz_locs=xyz_locs,
        pad_fraction=pad_fraction,
        points_x=xyz_locs[:, 0],
        points_y=xyz_locs[:, 1],
        points_z=xyz_locs[:, 2],
    )
    
        
    if do_batch_molecules:

        # Sub-select molecules in the current chunk
        xyz_subimg_locs = xyz_locs[img_mol_inds, :]
        x_locs = xyz_subimg_locs[:,0]
        y_locs = xyz_subimg_locs[:,1]
        z_locs = xyz_subimg_locs[:,2]

        # Expand for broadcasting
        x_locs = jnp.expand_dims(x_locs,(0,1,2))
        y_locs = jnp.expand_dims(y_locs,(0,1,2))
        z_locs = jnp.expand_dims(z_locs,(0,1,2))
        jxx = jnp.expand_dims(jxx,(3))
        jyy = jnp.expand_dims(jyy,(3))
        jzz = jnp.expand_dims(jzz,(3))
        
        # Push to GPU for minor speedup
        x_locs = device_put(x_locs)
        y_locs = device_put(y_locs)
        z_locs = device_put(z_locs)
        jxx = device_put(jxx)
        jyy = device_put(jyy)
        jzz = device_put(jzz)

        sub_img.values = jax_broadcast_reduce_pdf_fn(x_locs,y_locs,z_locs,jxx,jyy,jzz)

    else:

        # In some cases we might want to generate molecules in batches.  E.g. if the jax_broadcast_reduce_pdf_fn
        # fails due to OOM errors.  However, it is almost always better to just reduce your batch-size instead.
        
        def batch_molecules(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

    
        #print(f'Breaking down molecules into {np.ceil(len(img_mol_inds)/n_molecules_per_batch)} batches')
        for i,mol_inds in enumerate(batch_molecules(img_mol_inds,n_molecules_per_batch)):

            xyz_subimg_locs = xyz_locs[mol_inds, :]
            x_locs = xyz_subimg_locs[:,0]
            y_locs = xyz_subimg_locs[:,1]
            z_locs = xyz_subimg_locs[:,2]
            render_vmap = jax.vmap(jax_broadcast_pdf_fn,in_axes=(0,0,0,None,None,None),out_axes=(0))
            sub_img.values += render_vmap(x_locs,y_locs,z_locs,jxx,jyy,jzz).sum(axis=0)

        
    return sub_img

def generate_pdfs(n_pdfs):
    for k,v in locals().items():
        print(k,v)
    xyz_locs = jax.random.uniform(key,minval=coords_min,maxval=coords_max,shape=(n_pdfs,3))
    return xyz_locs


def test():
    ####################################

    do_batch_molecules = False
    n_molecules_per_batch = 100

    coords_min = -10
    coords_max = 10

    coord_size = 512
    chunk_size = 128

    n_pdfs = int(1e6)

    arr_dtype = np.float32

    pad_fraction = 0

    from jax import random
    key = random.PRNGKey(0)

    # Expected memory output.
    bytes_per_voxel = 32

    n_image_voxels = coord_size**3
    mem_per_image = sizeof_fmt(bytes_per_voxel * n_image_voxels, suffix="B")

    n_subimage_voxels = n_molecules_per_batch * chunk_size**3
    n_batch_voxels = n_molecules_per_batch * n_subimage_voxels

    mem_per_subimage = sizeof_fmt(bytes_per_voxel*n_subimage_voxels, suffix="B")
    mem_per_subimage_batch = sizeof_fmt(n_molecules_per_batch * bytes_per_voxel*n_subimage_voxels, suffix="B")
    mem_peak = sizeof_fmt(2*bytes_per_voxel * n_image_voxels, suffix="B")

    n_subimages = int(np.ceil(coord_size/chunk_size) ** 3)


    from jax.lib import xla_bridge
    print(xla_bridge.get_backend().platform)
    print()
    print(f'Expected number of pdfs {n_pdfs:.1g}')
    print(f'Expected number of subimages from xarray chunking: {n_subimages:.0f}')
    print(f'Approx number of molecules per subimage = {n_pdfs / n_subimages:.0f}')
    print(f'Approx number of batches per subimage = {n_pdfs / n_subimages / n_molecules_per_batch :.0f}')
    print()
    print(f'Expect vmap batch size of {n_batch_voxels:g} as (molecules={n_molecules_per_batch},x={chunk_size},y={chunk_size},z={chunk_size})')
    print()
    print(f'Expect image size of (x={coord_size},y={coord_size},z={coord_size}) = {mem_per_image}')
    print(f'Expect {n_subimages} subimages like (x={chunk_size},y={chunk_size},z={chunk_size}) = {mem_per_subimage}')
    print(f'Expected max memory for each subimage during vmap:',mem_per_subimage)
    print(f'Expected peak memory during final + working:',mem_peak)
    print()

    # ##############################

    chunks = (chunk_size,chunk_size,chunk_size)

    print('Initializing...')

    final_arr = initialize_image(coords_min,coords_max,coord_size,chunk_size,verbose=True)
    display(final_arr)

    print('\nBeginning renders')
    for i in range(3):
        
        print(f'\n---- {i} ----')
        xyz_locs = generate_pdfs(n_pdfs)
        #xyz_locs = device_put(xyz_locs)
        print('Rendering...',i)
        working_arr = xr.zeros_like(final_arr)
        working_arr = working_arr.map_blocks(render_subimage,kwargs={'xyz_locs':xyz_locs},template=working_arr)
        %time working_arr = working_arr.compute().chunk(chunks=chunks)
        final_arr = (final_arr + working_arr).compute().chunk(chunks=chunks)
        display(final_arr)
        
    fig,axs = plt.subplots(1,3,figsize=(10,30))
    axs[0].imshow(final_arr.sum(axis=0))
    axs[1].imshow(final_arr.sum(axis=1))
    axs[2].imshow(final_arr.sum(axis=2))
    plt.show()
