def save_to_zarr(ds,filename:str,variable=None,delete_existing_data=False,**kwargs):
    '''
    
    Supports:
    - Direct save of a dataset
    - Direct save of a DataArray, as a dataset with variable=variable
    - Replace subset of a DataArray within the main dataset
    - Append to one DataArray, and extend other dataArrays that share the same dimension to avoid errors.
    
     Persistence mode: 
         “w-” means create (fail if exists); [DEFAULT]
         “w” means create (overwrite if exists); 
         “a” means override existing variables (create if does not exist); 
         “r+” means modify existing array values only (raise an error if any metadata or shapes would change). 
         The default mode is “a” if append_dim [OR target_variable] is set. 
         Otherwise, it is “r+” if region is set and w- otherwise.
    '''
    from pathlib import Path
    zarr_path = Path(filename)
    
    print(kwargs)
    
    if type(ds) not in [xr.Dataset, xr.DataArray]:
        raise TypeError('Input data **ds** must be either a DataArray or a Dataset')

    
    elif type(ds) == xr.Dataset:
        print(f'Updating dataset at {zarr_path} with kwargs: {kwargs}')
        if delete_existing_data:
            kwargs['mode'] = 'w'
        ds.to_zarr(zarr_path,**kwargs)

    elif type(ds) == xr.DataArray:
        
        if variable is None:
            raise ValueError('Input is DataArray, but the group (i.e. variable name) is not defined.')
        else:
            print(f'Updating dataset variable **{variable}** : zarr_path at {zarr_path}, kwargs: {kwargs}')

        ds_update = ds.to_dataset(name=variable)                
        ds_ondisk = xr.open_zarr(zarr_path)

        try:
            xr.align(ds_update[variable],ds_ondisk[variable],join='exact')
            input_is_aligned_with_disk = True
        except:
            input_is_aligned_with_disk = False

        if input_is_aligned_with_disk:
            
            print(f'Input variable <{variable}> matches shape on disk - updating existing values.')
            kwargs['mode'] = 'r+'
            ds_update.to_zarr(zarr_path,**kwargs)

        else:
            
            # Test if the updated variable is a subset of the variable on disk.
            try:
                ds_update_aligned,_ = xr.align(ds_update[variable],ds_ondisk[variable],join='inner')
                xr.align(ds_update[variable],ds_update_aligned,join='exact') # FAILS IF NOT A SUBSET
                input_is_subset_of_disk = True
            except:
                input_is_subset_of_disk = False

            # By default we have to align to all regions in the target array on disk.
            region_dims = list(ds_ondisk[variable].coords.keys())
            
            if input_is_subset_of_disk:   
                
                print(f'Input is subset of disk - updating  existing values in region') 
                
                def region_slices(region_dims,disk_coords,update_coords):
                    '''
                    Returns a slice() object represengint indices into disk_coords for the subregion defined in update_coords.
                    This becomes the region= keyowrd to .to_zarr()

                    If both disk_coords and update_coords happen to have zero-indexed int-typed coordinates, this one-liner also works.     
                    return {dim:slice( int(coord[0]),int(coord[-1]+1)) for dim,coord in update_coords.items()}
                    '''
                    slice_dict = {key:None for key in region_dims}
                    for dim in region_dims:
                        disk_coord = np.array(disk_coords[dim])
                        update_coord = np.array(update_coords[dim])      
                        first = np.nonzero(disk_coord == update_coord[0])[0][0]
                        last = np.nonzero(disk_coord == update_coord[-1])[0][0] + 1
                        slice_dict[dim] = slice(first,last)
                    return slice_dict

            
                # Get slices for overlapping subregion
                kwargs['region'] = region_slices(region_dims, ds_ondisk[variable].coords, ds_update[variable].coords)            
                # Write to disk.
                ds_update.to_zarr(zarr_path,**kwargs)
            
                # By default we have to align to all regions in the target array on disk.
                region_dims = list(ds_ondisk[variable].coords.keys())

            elif not(input_is_subset_of_disk) and ('append_dim' in kwargs.keys()):

                print(f'Updating region with slices and appending to dim <{kwargs["append_dim"]}>:') 

                # If there are other variables in the ds_on_disk that share the 
                # same dimension as *append_dim*, then we have to align and extend them to 
                # make sure that they match with the updated variable.
                for name,var in ds_ondisk.data_vars.items():
                    print('Testing alignment for',name,var)
                    if name in ds_update.data_vars:
                        continue
                    elif kwargs['append_dim'] in var.coords:
                        print(f'Aligning variable {name} with the target variable')
                        ds_update[name], _ = xr.align( var, ds_update, join='outer')
                
                #kwargs['region'] = region_slices(region_dims, ds_ondisk[variable].coords, ds_update[variable].coords)            
                ds_update.to_zarr(zarr_path,**kwargs)
            
            else:
                raise ValueError("Something is wrong - input didn't match teh size of the array on disk, wasn't as subset of the array on disk.  \
                                 This implies an append call, but append_dims was not set.")
             