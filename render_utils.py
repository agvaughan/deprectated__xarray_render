import netCDF4 # For xarray
import h5netcdf # For xarray

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import xarray as xr
import dask.array as da
import numpy as np
from dask.diagnostics import ProgressBar

  
@jax.jit
def jit_generate_poisson_emissions(emission_image, channel_amplitude,output_dtype='Int16'):
    '''
        Renders poisson (photon-like) emission from a base image of  emission intensities (emission_image) 
        across a set of channels (channel_amplitude)

        Image shape is broadcast to jax.random.poisson(lam)

        Inputs:
            emission_image : image of poisson intensities (i.e. lambda)
            channel_amplitude : channels to broadcast to, with 

        Outputs: jax.DeviceArray


    '''
    rate = emission_image * channel_amplitude
    # jax.random.poisson allows for broadcasting here.
    return jax.random.poisson(key,lam=rate,dtype=output_dtype)


def launch_block(intensities_subxr,channel_amplitude_xr,key):
    '''
    Block-wise launcher for arbitrary ufuncs.

    Here launching 
        jit_generate_poisson_emissions(intensities_subxr, channel_amplitude_xr)
    to render poisson emissions

    Inputs:
        intensities_subxr - xarray sub-array containing dask chunks + xarray dimensions
        channel_amplitude_xr - xarray containing channels * timepoints with  xarray dimensions
        key - jax.random.key [note cannot be reused - split within this subfunction]

    Outputs:
        Subimage with poisson intensities, as an xarray.DataArray

    '''

    _,key = jax.random.split(key)
    (aligned_intensities_subxr,_) = xr.broadcast(intensities_subxr,channel_amplitude_xr)
    output = xr.DataArray((xr.apply_ufunc(jit_generate_poisson_emissions,
                                            aligned_intensities_subxr.values,
                                            channel_amplitude_xr.values)),
                            dims=dims)
    return output





def test():


    def print_dims(name):
        ar = eval(name)
        print(name,type(ar))
        print('\tDim Sizes:   ',{dim:shape for dim,shape in zip(ar.dims,ar.shape)})
        if ar.chunks is not None:
            print('\tChunk Sizes: ',{dim:shape for dim,shape in zip(ar.dims,[chunks[0] for chunks in ar.chunks])})
        display(ar)

    def plot_xyz(xr_xyz):
        xr_xyz.sum('z').plot.imshow()
        plt.show()
        
    def plot_xyzct(xr_xyzct):
        xr_xyzct.sum('z').plot(x='x',y='y',col='c',row='t',figsize=(20,20))
        plt.show()
        
    key = jax.random.PRNGKey(0)

    n         = 1024
    chunk_xyz = 256
    chunk_ct  = 4

    x = n
    y = n
    z = n
    c = 4
    t = 4

    poisson_offset = 200

    working_type = 'int16'
    output_dtype = 'int16'

    ###########################################################
    chunks_ct = (chunk_ct,)*2
    chunks_xyz = (chunk_xyz,)*3
    chunks_xyzct = (*chunks_xyz,*chunks_ct)
    dims = ['x','y','z','c','t'] 
    chunk_dict = {dim:chunk for dim,chunk in zip(dims,chunks_xyzct)}

    # Colormap
    colormap = np.eye(c)
    channel_amplitude = colormap
    channel_amplitude_xr = (xr.DataArray(channel_amplitude,dims=['c','t'])
                            .chunk(chunks_ct))

    # Emission intensity
    xyz = (np.expand_dims(np.linspace(0,1,x),(1,2)) * 
        np.expand_dims(np.linspace(0,1,y),(0,2)) * 
        np.expand_dims(np.linspace(0,1,z),(0,1)))
    emission_xr = (xr.DataArray(da.array(xyz),dims=['x','y','z']).chunk(chunks_xyz))
    emission_xr = (emission_xr * poisson_offset).persist()
    del xyz

    # Retype
    emission_xr = emission_xr.astype(working_type)
    channel_amplitude_xr = channel_amplitude_xr.astype(working_type)



    # Info
    print_dims('emission_xr')
    plot_xyz(emission_xr)
    print_dims('channel_amplitude_xr')
    channel_amplitude_xr.plot.imshow()
    plt.show()

if __name__ == "__main__":
    test()

    with ProgressBar():
        output = (emission_xr.map_blocks(launch_block,
                                            args = (channel_amplitude_xr,key)
                                        ).persist())
    print_dims('output')
    plot_xyzct(output)