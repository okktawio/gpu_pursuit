# -*- coding: utf-8 -*-
import pyopencl as cl
import numpy as np
import sys

class gpu_base:
    """
    Class that performs matching and basis pursuit on space--time cubes
    """
    def __init__(self, valuecube,
                     timecube,
                     nyquist = 23. / 2.,
                     maxitems = 256,
                     missingvalue = -9999.9,
                     maxatoms = 20,
                     alpha = 1e-6,
                     tolerance = 1e-10,
                     gradient_tolerance = 1e-9,
                     bfgs_iterations = 40):
        """
        Initialize the class, compile the opencl code, upload data, and initialize the params
        """
        #number of local threads in the opencl code (AMD=256, NVIDIA=1024)
        #Warning: using more than 256 threads might overflow the local memory
        self.cube = valuecube
        self.time = timecube
        self.timelimits = [timecube.max(), timecube.min()]
        self.maxitems = maxitems
        self.nyquist = nyquist
        self.missingvalue = missingvalue
        self.maxatoms = maxatoms
        self.maxparams = maxatoms * 4
        self.calculate_error_params = False
        self.center = True
        self.alpha = alpha
        self.tolerance = tolerance
        self.gradient_tolerance = gradient_tolerance
        self.bfgs_iterations = bfgs_iterations
        self.init_cl()
        self.upload_data()
        self.init_params()

    def init_cl(self, CPU = False, device = 1, source_file = "gpu_pursuit.cl"):
        """
        Compile the OpenCL code
        """
        
        platform = cl.get_platforms()
        
        #select device type
        if CPU: dvtype = cl.device_type.CPU
        else:
                dvtype = cl.device_type.GPU
                device = 0

        #initialize context
        self.ctx = cl.Context(devices = platform[device].get_devices(device_type = dvtype))

        #read the cl code from an file
        clcode =  open(source_file).read()

        #init some macros
        #max length of auxiliary vectors 
        clcode = clcode.replace("__lmax__","%d"%self.maxitems)
        #nyquist frequency
        clcode = clcode.replace("__nyquist__","%12.9ff"%self.nyquist)
        #wavelength of the nyquist frequency
        clcode = clcode.replace("__lnyquist__","%12.9ff"%(1./self.nyquist))
        #timelimits
        clcode = clcode.replace("__tmax__","%12.9ff"%self.timelimits[0])
        clcode = clcode.replace("__tmin__","%12.9ff"%self.timelimits[1])
        #error value
        clcode = clcode.replace("__missing__","%f"%self.missingvalue)
        #length of time series
        #is the last axis of the space-time cube
        clcode = clcode.replace("__len_t_","%d"%self.time.shape[-1])
        #length of the vector with parameters (4 * max atoms + 1 (mean))
        clcode = clcode.replace("__nparams__","%d"%(self.maxparams))
        #length of the vector with parameters (4 * max atoms + 1 (mean))
        clcode = clcode.replace("__alpha__","%f"%(self.alpha))
        clcode = clcode.replace("__tol__","%f"%(self.tolerance))
        clcode = clcode.replace("__gtol__","%f"%(self.gradient_tolerance))
        clcode = clcode.replace("__iiter__","%f"%(self.bfgs_iterations))
        #compiling the cl code
        self.prg = cl.Program(self.ctx, clcode).build(options = "", cache_dir=False)
        #keeping the CL variables into the object
        self.queue = cl.CommandQueue(self.ctx)
        self.mf = cl.mem_flags

    def upload_data(self):
        #matrix with data
        self.cl_cube    = cl.Buffer(self.ctx, self.mf.READ_ONLY  | self.mf.COPY_HOST_PTR, hostbuf = self.cube)
        #matrix with time data
        self.cl_time    = cl.Buffer(self.ctx, self.mf.READ_ONLY  | self.mf.COPY_HOST_PTR, hostbuf = self.time)

    def download_params(self):
        """
        Download from the GPU memory the parameters
        returns the matrix with parameters, the matrix with number of parameters
        """
        cl.enqueue_copy(self.queue, self.params, self.cl_params)
        cl.enqueue_copy(self.queue, self.nparams, self.cl_nparams)
        if self.calculate_error_params:
            cl.enqueue_copy(self.queue, self.errparams, self.cl_errparams)
            return self.params, self.nparams, self.errparams
        return self.params, self.nparams

    def init_errparams(self):
        """
        generates the matrix which contains the standard deviation of calculated parameters.
        """
        if self.calculate_error_params:
            self.errparams        = np.zeros_like(self.params)
        else:
            self.errparams        = np.zeros((1, 1, 1), np.float32)
        self.cl_errparams    = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf = self.errparams)
    
    def init_params(self):
        #init parameters matrix in zero
        self.params         = np.zeros((self.cube.shape[0], self.cube.shape[1], self.maxatoms, 4), np.float32)
        #upload
        self.cl_params      = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf = self.params)
        #init matrix with number of parameters in zero
        self.nparams        = np.zeros((self.cube.shape[0], self.cube.shape[1]), np.int32)
        self.cl_nparams     = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf = self.nparams)
        #init matrix with number of parameters in zero
        self.init_errparams()

        

class gpu_pursuit(gpu_base):
    def matching_pursuit(self,
                             center = True,
                             iterations = 4,
                             minvalids = 100,
                             objective_residuals = 1e-6,
                             objective_atoms = -1,
                             return_residuals = False):
        '''
        Matching Pursuit
        '''
        self.center = center
        self.algorithm  = 0
        self.iterations = iterations
        self.minvalids  = minvalids
        self.objective_residuals = objective_residuals
        self.objective_atoms = objective_atoms
        self.calculate_error_params = 0
        self.return_residuals = return_residuals
        self.fit_pursuit()
        self.download_params()
        
    def basis_pursuit(self,
                          center = True,
                          iterations = 4,
                          minvalids = 100,
                          objective_residuals = 1e-6,
                          objective_atoms = -1,
                          calculate_error_params = 1,
                          return_residuals = False):
        '''
        Basis Pursuit using Maximum likelihood as utility function
        '''
        self.center = center
        self.algorithm  = 1
        self.iterations = iterations
        self.minvalids  = minvalids
        self.objective_residuals = self.cube.shape[2] * np.log(objective_residuals/self.cube.shape[2])
        self.objective_atoms = objective_atoms
        self.calculate_error_params = calculate_error_params
        self.return_residuals = return_residuals
        self.init_errparams()
        self.fit_pursuit()
        self.download_params()
        
    def basis_pursuit_aic(self,
                          center = True,
                          iterations = 4,
                          minvalids = 100,
                          objective_residuals = 1e-6,
                          objective_atoms = -1,
                          calculate_error_params = 1,
                          return_residuals = False):
        '''
        Basis Pursuit using Akaike Information Criterion as utility function
        '''
        self.algorithm  = 2
        self.iterations = 0
        self.minvalids  = minvalids
        self.objective_residuals = self.cube.shape[2] * np.log(objective_residuals/self.cube.shape[2])
        self.objective_atoms = objective_atoms
        self.calculate_error_params = calculate_error_params
        self.return_residuals = return_residuals
        self.init_errparams()
        self.fit_pursuit()
        self.download_params()
                
    def basis_pursuit_aicc(self,
                          center = True,
                          iterations = 4,
                          minvalids = 100,
                          objective_residuals = 1e-6,
                          objective_atoms = -1,
                          calculate_error_params = 1,
                          return_residuals = False):
        '''
        Basis Pursuit using Corrected Akaike Information Criterion as utility function
        '''
        self.center = center
        self.algorithm  = 3
        self.iterations = 0
        self.minvalids  = minvalids
        self.objective_residuals = self.cube.shape[2] * np.log(objective_residuals/self.cube.shape[2])
        self.objective_atoms = objective_atoms
        self.calculate_error_params = calculate_error_params
        self.return_residuals = return_residuals
        self.init_errparams()
        self.fit_pursuit()
        self.download_params()
        
    def basis_pursuit_bic(self,
                          center = True,
                          iterations = 4,
                          minvalids = 100,
                          objective_residuals = 1e-6,
                          objective_atoms = -1,
                          calculate_error_params = 1,
                          return_residuals = False):
        '''
        Basis Pursuit using Bayesian Information Criterion as utility function
        '''
        self.center = center
        self.algorithm  = 4
        self.iterations = 0
        self.minvalids  = minvalids
        self.objective_residuals = self.cube.shape[2] * np.log(objective_residuals/self.cube.shape[2])
        self.objective_atoms = objective_atoms
        self.calculate_error_params = calculate_error_params
        self.return_residuals = return_residuals
        self.init_errparams()
        self.fit_pursuit()
        self.download_params()
        
    def fit_pursuit(self):
        '''
        Performs pursuit
        '''
        self.prg.gpu_pursuit(
            self.queue,
            (self.maxitems * self.cube.shape[0] * self.cube.shape[1],),
            (self.maxitems,),
            self.cl_cube,
            self.cl_time,
            self.cl_params,
            self.cl_errparams,
            self.cl_nparams,
            np.uint64(np.random.randint(0, np.iinfo(np.int64).max)),
            np.int32(self.iterations),
            np.int32(self.center),
            np.int32(self.minvalids),
            np.float32(self.objective_residuals),
            np.int32(self.objective_atoms),
            np.int32(self.algorithm),
            np.int32(self.return_residuals))

