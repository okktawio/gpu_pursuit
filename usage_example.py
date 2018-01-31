from gpu_pursuit import *
import numpy as np

if __name__ == "__main__":
    np.set_printoptions(linewidth = 360, precision = 4, threshold='nan', suppress=True)
    np.set_printoptions(suppress = "True")

    #space time modis ndvi cube
    cube   = np.load("pilca_cube.array"%lugar)
    #time release of each modis ndvi image
    time = np.load("pilca_time.array"%lugar)[:cube.shape[2]]
    time = time.astype(np.float32) / 365.25
    #date of each pixel datum
    days   = np.load("pilca_DAY.array"%lugar)
    #transform to a float32 array
    cube = cube.astype(np.float32) / 10000.
    cube = np.float32(cube * (cube > 0) - 99999 * (cube <= 0))
    cube = np.array(cube[:,:,:days.shape[2]], copy = 1)
    time = time[:cube.shape[2]]

    #transform from days to a year units, and correct the date
    nt   = np.float32((days / 365.25 + np.floor(time)) * (days >= 0)  + time * (days < 0))
    nt[:,:,19::23] = (nt[:,:,19::23] + (days[:,:,19::23] < 353) * (days[:,:,19::23] >= 0))

    gpp = gpu_pursuit(cube, nt, alpha = 1e-4)
    
    #Matching Pursuit
    gpp.matching_pursuit(objective_residuals = 10., iterations = 128, minvalids = 50)
    params, nparams = gpp.download_params()

    print nparams
    print params

    #Basis Pursuit
    gpp.basis_pursuit_aicc( minvalids = 50, objective_atoms = 20, calculate_error_params = 1)
    params, nparams, sddparams = gpp.download_params()

    print nparams
    print params
    print sddparams
                    


