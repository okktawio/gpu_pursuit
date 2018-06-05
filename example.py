from gpu_pursuit import *
import numpy as np

if __name__ == "__main__":
    np.set_printoptions(linewidth = 360, precision = 4, threshold='nan', suppress=True)
    np.set_printoptions(suppress = "True")

    #space time modis ndvi cube
    data = np.load("pilca.npz")
    cube = data["cube"]
    #time release of each modis ndvi image
    time = cube = data["time"]
    time = time.astype(np.float32) / 365.25
    #date of each pixel datum
    days   = cube = data["DAY"]
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
    #returns a four dimension matrix with the estimated parameters (params),
    #and a bidimensional matrix with the number of parameters per pixel
    params, nparams = gpp.download_params()

    #Basis Pursuit
    gpp.basis_pursuit_aicc( minvalids = 50, objective_atoms = 20, calculate_error_params = 1)
    #returns a four dimension matrix with the estimated parameters (params),
    #a bidimensional matrix with the number of parameters per pixel
    #and a four dimension matrix with the estimation error of each parameter
    params, nparams, sddparams = gpp.download_params()
    
    #each params matrix contains a per pixel list of the parameters of the estimated gabor atoms
    print params.shape #it must be (42, 60, 20, 4)
    #being the first itwo ndexes, the position of the pixel in the matrix
    #the third, the atom, and the last, the four parameters of each atom
    #the parameters are
    # 0 - time center of each atom (u),
    # 1 - standard deviation of each gaussian window (s)
    # 2 - frequency of the cosine function (f)
    # 3 - amplitude (a)
    #so the pixel [10,10] must contain
    '''
    In [44]: print params[10,10]
    #u    s       f      a
[[  3.88  34.09  -0.05   0.43]
 [  4.58  26.92  -0.99  -0.31]
 [  8.13  13.34   0.12   0.35]
 [ 13.83  33.48  -1.02   0.15]
 [  6.35  15.58  -0.27   0.19]
 [ 15.09  17.29   0.19   0.16]
 [ 12.95  17.68  -0.75  -0.16]
 [ 16.88  25.59  -2.01   0.14]
 [ 11.69   5.41  -0.45  -0.26]
 [ 17.98  30.7   -0.87   0.08]
 [ 11.66   4.49  -1.19  -0.2 ]
 [  7.46  25.93  -0.33   0.09]
 [  9.65  27.99   1.58  -0.08]
 [ 11.05  11.33   0.66   0.1 ]
 [ 15.05  23.59  -0.93  -0.08]
 [  1.37  17.44   0.12  -0.1 ]
 [ 11.56  28.95   1.85  -0.07]
 [  2.68  16.05   5.36  -0.07]
 [  1.16  32.13   2.4   -0.06]
 [  7.39  26.56 -10.01   0.06]]
    '''
