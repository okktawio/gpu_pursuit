#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#error "Double precision floating point not supported by OpenCL implementation."
#endif

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics: enable

//minimum double precision value
#define min64f (1.f/18446744073709551615.f)

#define infinity 3.40e38
#define _minutility_ 0.001
#define _lmax_ __lmax__
#define _nyquist_ __nyquist__
#define _lnyquist_ __lnyquist__
#define _tmax_ __tmax__
#define _tmin_ __tmin__
#define _len_t __len_t_
#define _nparams_ __nparams__
#define _missing_ __missing__

#define index_params(ip, jp) gid * _nparams_  + 4 * ip + jp

///auxiliary routines *******************************************************
float local_sum_f(local float vector_data[], uint len)
{
  //sum a local memory floating point memory vector 
  uint i = get_local_id(0);
  uint pot2 = 0;
  while((len >> pot2) > 1) ++pot2;
  if(pot2 > 0)
    {
      uint max2 = 1 << pot2;
      uint dif2 = len - max2;
      uint min2 = max2 - dif2;
      if(i >= min2 && i < max2)
	vector_data[i] += vector_data[i + dif2];
      barrier(CLK_LOCAL_MEM_FENCE);
      while(pot2 > 0)
	{
	  --pot2;
	  max2 = 1 << pot2;
	  if( i < max2)
	    vector_data[i] += vector_data[i + max2];
	  barrier(CLK_LOCAL_MEM_FENCE);
	}
   }
  barrier(CLK_LOCAL_MEM_FENCE);
  return vector_data[0];
}

void copy_cube_vector(global float * icubo, local float * seriet)
{
  //copy data from global memory 3d space time cube
  //to a local memory vector
  int jj;
  uint gid = get_group_id(0);
  uint   i = get_local_id(0);
  for(int j = 0; j < _len_t; j += _lmax_)
    {
      jj = i + j;
      if(jj < _len_t)
	seriet[jj] = icubo[gid * _len_t + jj];
    }
}

void copy_vector_cube(global float * icubo, local float * seriet)
{
  //copy data from global memory 3d space time cube
  //to a local memory vector
  int jj;
  uint gid = get_group_id(0);
  uint   i = get_local_id(0);
  for(int j = 0; j < _len_t; j += _lmax_)
    {
      jj = i + j;
      if(jj < _len_t)
	icubo[gid * _len_t + jj] = seriet[jj];
    }
}

float count_valid(local float serie[], local float aux[])
{
  //counts valid data
  uint i = get_local_id(0);
  int jj;
  if(i < _lmax_)
    aux[i] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int j = 0; j < _len_t; j += _lmax_)
    {
      jj = i + j;
      if(jj < _len_t)
	aux[i] += (float) (serie[jj] > _missing_);
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  float smm = local_sum_f(aux, _lmax_);
  barrier(CLK_LOCAL_MEM_FENCE);
  return smm;
}


float center_series(local float * serie, local float * aux, float nnn)
{
  //remove the mean of a local memory vector
  //serie: vector with the data series
  //aux. auxiliary vector with the same length as the number of threads
  //nnn: numbero of valid data
  uint i   = get_local_id(0);
  int jj;
  float mean;
  aux[i] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  //sumatoria
  for(int j = 0; j < _len_t; j += _lmax_)
    {
      jj = i + j;
      if(jj < _len_t)
	//sum only valid data
	if(serie[jj] > _missing_)
	    aux[i] += (float) serie[jj];
    }

  barrier(CLK_LOCAL_MEM_FENCE);
  mean = local_sum_f(aux, _lmax_) / nnn;
  
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int j = 0; j < _len_t; j += _lmax_)
    {
      jj = i + j;
      if(jj < _len_t)
          if(serie[jj] > _missing_) serie[jj] -= mean;
    }
  barrier(CLK_LOCAL_MEM_FENCE);
  return mean;
}

//random numbers*******************************************************************
//xorshiftstar uniform random number generator CITA PENDIENTE

inline
ulong xorshift64star(ulong x)
{
  //x = (ulong) x;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  return x * 2685821657736338717L;
}

//return uniformly distributed random numbers between 0 and 1
inline
float uniform(ulong i){
  return i * min64f;
}

//return uniformly distributed random numbers between min and max
float uniform_rng_p(ulong * seed, float min, float max)
{
  seed[0] = xorshift64star(seed[0]);
  return uniform(seed[0]) * (max - min) + min;
}

///end random numbers*********************************************************

float3 choose_best(local float3 vector[])
{
  //choose the vector with best utility
  uint i = get_local_id(0);
  uint pot2 = 8;
  uint max2;
  while(pot2 > 0)
    {
      --pot2;
      max2 = 1 << pot2;
      if(i < max2)
	{
	  if(vector[i].y > vector[i + max2].y) vector[i + max2] = vector[i];
	  else                                 vector[i] = vector[i + max2];
	}
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  barrier(CLK_LOCAL_MEM_FENCE);
  return vector[0];
}

///end auxiliary routines****************************************************************************************

///gabor functions ****************************************************************************************
float gabor(float t, float u, float s, float w)
{
  //gabor atom:
  //t: the time at which the gabor function is evaluated
  //u: center of the gabor atom
  //s: the standard deviation of the normal function
  //w: frequency of the periodic function
  return native_exp((float) -M_PI * pown((float) ((t - u) / s), 2)) * native_cos((float) (2 * M_PI * (t - u) * w));
}

//on each thread
//loop over the vector to calculate the inner product between the gabor atoms and the data series
float2 gabor_inner(local float * time, local float * data, float u, float s, float w)
{
  float ge = 0;
  float2 inner = (float2) (0.0f, 0.0f);

  for(int i = 0; i < _len_t; ++i)
    {
      if(data[i] > _missing_)
	{
	  ge = gabor(time[i], u, s, w);
	  inner.x += ge * data[i];
	  inner.y += ge * ge;
	}
    }
  return inner;
}

//sum of absolute residuals on each work item
float gabor_sum_residuals(local float * time, local float * data, float u, float s, float w, float a)
{
  float residuals = 0;

  for(int i = 0; i < _len_t; ++i)
      if(data[i] > _missing_)
	residuals += fabs(data[i] - gabor(time[i], u, s, w) * a);
  return residuals;
}

//sum of residuals on each work item
float gabor_sum_squares(local float * time, local float * data, float u, float s, float w, float a)
{
  float residuals = 0;

  for(int i = 0; i < _len_t; ++i)
      if(data[i] > _missing_)
	residuals += pown(data[i] - gabor(time[i], u, s, w) * a, 2);
  return residuals;
}

//sum of absolute residuals on each work group
float gabor_sum_residuals_local(local float * time, local float * data_series, float4 atom, local float * aux, int method)
{
  uint i = get_local_id(0);
  int jj;

  //initializes aux vector to zero
  aux[i] = 0;
  
  for(int j = 0; j < _len_t; j += _lmax_)
    {
      jj = i + j;
      if(jj < _len_t)
	if(data_series[jj] > _missing_)
	  {
	    //put the residuals back into data series vector
	    data_series[jj] -= gabor(time[jj], atom.x, atom.y, atom.z) * atom.w;
	    //sum to the aux vector
	    switch(method)
	      {
	      case 0:
		//absolute diferences
		aux[i] += fabs(data_series[jj]); break;
	      default:
		//squared differences
		aux[i] += pown(data_series[jj], 2); break;
	      }
	  }
    }
  barrier(CLK_LOCAL_MEM_FENCE);
  float smm = local_sum_f(aux, _lmax_);
  return smm;
}



float4 init_atom_p(ulong * seed)
{
  float center_time  =  uniform_rng_p(seed, _tmin_, _tmax_);
  float window_width = _lnyquist_ * 4 + (acos(uniform_rng_p(seed, 0, 1)) / M_PI) * (_tmax_ - _tmin_) * 4;
  float frequency    = 2 * _nyquist_ * (acos(uniform_rng_p(seed, -1, 1)) / M_PI) - _nyquist_;
  float4 atom = (float4) (center_time, window_width, frequency, 0.0);
  return atom;
}

///end of gabor functions ***************************************************************************************************


inline float16 eye16()
{
  return (float16) (1.0, 0.0, 0.0, 0.0,
		    0.0, 1.0, 0.0, 0.0,
		    0.0, 0.0, 1.0, 0.0,
		    0.0, 0.0, 0.0, 1.0);
}

inline float4 dot_16_4(float16 M, float4 V)
{
  return (float4) (M.s0 * V.x + M.s1 * V.y + M.s2 * V.z + M.s3 * V.w,
		   M.s4 * V.x + M.s5 * V.y + M.s6 * V.z + M.s3 * V.w,
		   M.s8 * V.x + M.s9 * V.y + M.sa * V.z + M.sb * V.w,
		   M.sc * V.x + M.sd * V.y + M.se * V.z + M.sf * V.w);
}

inline float16 outer4(float4 x, float4 y)
{
  return (float16) (x.x * y.x, x.x * y.y, x.x * y.z, x.x * y.w,
		    x.y * y.x, x.y * y.y, x.y * y.z, x.y * y.w,
		    x.z * y.x, x.z * y.y, x.z * y.z, x.z * y.w,
		    x.w * y.x, x.w * y.y, x.w * y.z, x.w * y.w);
}

inline float dot4(float4 x, float4 y)
{
  return (x.x * y.x + x.y * y.y + x.z * y.z + x.w * y.w);
}


inline float norm4(float4 g)
{
  return native_sqrt(g.x * g.x + g.y * g.y + g.z * g.z + g.w * g.w);
}


float4 derivative_atom(local float time[], local float data[], float4 atom)
{
  float fit0;
  float4 derivative;
  //first derivative of the fitting function for a given atom
  //in terms of residuals
  fit0 = gabor_sum_squares(time, data, atom.x, atom.y, atom.z, atom.w);
  derivative = (float4) (gabor_sum_squares(time, data, atom.x + 1e-5, atom.y, atom.z, atom.w) - fit0,
			 gabor_sum_squares(time, data, atom.x, atom.y + 1e-5, atom.z, atom.w) - fit0,
			 gabor_sum_squares(time, data, atom.x, atom.y, atom.z + 1e-5, atom.w) - fit0,
			 gabor_sum_squares(time, data, atom.x, atom.y, atom.z, atom.w + 1e-5) - fit0);
  return derivative / (float4) 1e-5;
}

inline float checklimits(float4 atom)
{
  float penalization = 0;
  //penalizes for exceeding limits
  if(atom.z < -_nyquist_) penalization += (_nyquist_ - atom.z) * 100.0;
  if(atom.z >  _nyquist_) penalization += (atom.z - _nyquist_) * 100.0;
  if(atom.y < _lnyquist_ * 4) penalization += (_lnyquist_ * 4 - atom.y) * 10000.0;
  if(atom.x < _tmin_) penalization += (_tmin_ - atom.x) * 10;
  if(atom.x > _tmax_) penalization += (atom.x - _tmax_) * 10;
  return penalization;
}


inline float error_atom(local float time[], local float data[], float4 atom)
{
  float penalization = checklimits(atom);
  float err = gabor_sum_squares(time, data, atom.x, atom.y, atom.z, atom.w) + penalization;
  if(isnan(err) || isinf(err))  return 1e9;
  return err;
}

inline float deviance_atom(local float time[], local float data[], float4 atom, int n)
{
  float rss = error_atom(time, data, atom);
  float deviance = n * log(rss / n);
  return deviance;
}

float dif_like_atom(local float time[], local float data[], float4 atom, int n, int ix)
{
  float4 atomi = atom;
  switch(ix)
    {
    case 0: atomi.x -= 1e-5; break;
    case 1: atomi.y -= 1e-5; break;
    case 2: atomi.z -= 1e-5; break;
    case 3: atomi.w -= 1e-5; break;
    }
  float lk0 = deviance_atom(time, data, atomi, n);
  float lk1 = deviance_atom(time, data, atom , n);
  switch(ix)
    {
    case 0: atomi.x += 2e-5; break;
    case 1: atomi.y += 2e-5; break;
    case 2: atomi.z += 2e-5; break;
    case 3: atomi.w += 2e-5; break;
    }
  float lk2 = deviance_atom(time, data, atomi, n);
  barrier(CLK_LOCAL_MEM_FENCE);
  float dif0 = (lk1 - lk0) * 10000.;
  float dif1 = (lk2 - lk1) * 10000.;
  float dif2 = (lk1 - lk0) * 10000.;
  //barrier(CLK_LOCAL_MEM_FENCE);
  return 1.0/native_sqrt(-dif2);
}

float4 stdparams(local float time[], local float data[], float4 atom, int n)
{
  float4 std;
  std.x = dif_like_atom(time, data, atom, n, 0);
  std.y = dif_like_atom(time, data, atom, n, 1);
  std.z = dif_like_atom(time, data, atom, n, 2);
  std.w = dif_like_atom(time, data, atom, n, 3);
  return std;
}
  

//some macros for the bfgs algorithm
//number of parameters per atom
#define  N      4
//maximum number of bfgs iterations
#define _iiter_ __iiter__
//tolerance in 
#define _tol_   __tol__
#define _gtol_  __gtol__
#define _alpha_ __alpha__

inline float parabolic_min(float x0, float y0, float x1, float y1, float x2, float y2)
{
  return (((y0*((x1*x1) - (x2*x2))) + (y1*((x2*x2) - (x0*x0))) + (y2*((x0*x0) - (x1*x1))))
	  /(2.0*((y0*(x1 - x2)) + (y1*(x2 - x0)) + (y2*(x0 - x1)))));
}

float parabolic_linesearch(local float time[], local float data[], float4 atom, float4 s, int n)
{
  /*
    Parabolic line search routine
    it is simple and fast for fitting trigonometric functions
  */
  float a3 = 2 * _alpha_;
  float a2 = _alpha_;
  float a1 = 0;
  float e1 = deviance_atom(time, data, atom, n);
  float e2 = deviance_atom(time, data, atom + s * (float4) a2, n);
  float e3 = deviance_atom(time, data, atom + s * (float4) a3, n);
  float e0 = e1;
  //if the derivative at a1 is positive there is already in the minimum
  //so end the procedure
  if(e1 < e2 && e1 < e3) return 0;
  //now look for the interval in which there is the minimum
  while (e3 <= e2)
    {
      a1 = a2;
      e1 = e2;
      a2 = a3;
      e2 = e3;
      a3 *= 2;
      e3 = deviance_atom(time, data, atom + s * (float4) a3, n);
    }
  float a4 = parabolic_min(a1, e1, a2, e2, a3, e3);
  float e4 = deviance_atom(time, data, atom + s * (float4) a4, n);
  if(e2 < e4) return a2;
  return a4;
}

float8 bfgs_p(local float time[], local float data[], float4 atom, int n)
{
  /*
    Performs boundary chequed bfgs optimization, minimizing the -likelihood function on gabor atoms.
    It is based in a mixture of many routines found in the web, most notably the bfgs from 
    scipy and the bfgs routine from https://github.com/tazzben/EconScripts
   */
  float like = 0;
  //initial amplitude estimation
  float2 inner = gabor_inner(time, data, atom.x, atom.y, atom.z);
  atom.w = inner.x / inner.y;
  //init old atom with some extreme value
  float4 atom_old = atom + (float4) 1e9;
  //init the Hessian matrix
  float16 Hessian = eye16();
  //bfgs iterations
  uint   bfgs_i = 0;
  float4 gradient = derivative_atom(time, data, atom);
  
  while((norm4(gradient) > _gtol_) && (norm4(atom - atom_old) > _tol_) && (bfgs_i < _iiter_))	
    {
      atom_old = atom;
      ++bfgs_i;

      //calculate search direction
      float4 search = (float4) -dot_16_4(Hessian, gradient);
      //performing line search to calculate step length (alpha)
      float  alpha  = parabolic_linesearch(time, data, atom, search, n);
      
      if (alpha > 0)
	{
	  //updating atom with the new values
	  atom += search * (float4) alpha;

	  //saving previous gradient value
	  float4 gradientold = gradient;
	  //calculate new gradient for the updated atom
	  gradient = derivative_atom(time, data, atom);
	  //difference in gradient
	  float4 diffgradient = (gradient - gradientold) / (float4) alpha;
	  //difference in search direction
	  float  diffsearch = dot4(search, diffgradient);
	  if(diffsearch > 0)
	    {
	      //Update Hessian matrix
	      float4 Hdiff = dot_16_4(Hessian, diffgradient);
	      Hessian += outer4(search, search) * (float16) (dot4(search, diffgradient) + dot4(diffgradient, Hdiff)) /
		(float16) diffsearch * diffsearch - (outer4(Hdiff, search) + outer4(search, Hdiff))/(float16) diffsearch;
	    }
	}
      else break;
    }
  //
  //  returns the fitted atom (first 4 elements),
  //  and the square root of the inverse of the diagonal of the Hessian matrix
  return (float8)(atom.x, atom.y, atom.z, atom.w,
		  1.0/native_sqrt(Hessian.s0), 1.0/native_sqrt(Hessian.s5),
		  1.0/native_sqrt(Hessian.sa), 1.0/native_sqrt(Hessian.sf));
}

float4 select_atom(local float3 * vatoms, float utility, float errorf, float8 atom_i, local float4 * atomsf, local float4 * atomssd, int catoms)
{
  /*
    select the atom with best utility from a list
  */
  
  uint i = get_local_id(0);
  //thread number of each calculated atom
  vatoms[i].x = (float) i;
  //utility
  vatoms[i].y = utility;
  //error
  vatoms[i].z = errorf;
  barrier(CLK_LOCAL_MEM_FENCE);
      
  choose_best(vatoms);
  barrier(CLK_LOCAL_MEM_FENCE);
  uint best_index = (int) vatoms[0].x;

  barrier(CLK_LOCAL_MEM_FENCE);
  //copy the best parameters to the local memory
  if(i == best_index)
    {
      atomsf[catoms]  = atom_i.lo;
      atomssd[catoms] = atom_i.hi;
    }
  barrier(CLK_LOCAL_MEM_FENCE);
      
  //copy the best parameters back to private memory
  float4 best_params = atomsf[catoms];
  barrier(CLK_LOCAL_MEM_FENCE);
  return best_params;
}


float4 basis_pursuit_incrowd(local float * time,
			     local float * data_series,
			     local float * aux_vector,
			     local float3 * vatoms,
			     local float4 * atomsf,
			     local float4 * atomssd,
			     float oresiduals,
			     int oatoms,
			     ulong seed,
			     int method,
			     int nvalid)
{
  uint i   = get_local_id(0);
  uint gid = get_group_id(0);
  //parameters of the best atom
  float4 atom_i = (float4) (1.0f, 1.0f, 1.0f, 0.0f);
  float8 atom_sdd;
  if(method >= 2)
    {
      atom_i.w = 1;
      atom_i.z = 0;
    }
  //error at null atom
  float error0;
  error0 = deviance_atom(time, data_series, atom_i, nvalid);
  float utility = -1;
  int catoms = 0;
  //residuals
  float cresiduals;
  float information_criterion, information_criterion0 = 2 * error0;
  cresiduals = infinity;
  float errorf;
  while((catoms < oatoms) && (cresiduals > oresiduals))
    //in crowd algorithm
    {
      //parameters for the gabor atoms
      if(utility <= _minutility_) atom_i = init_atom_p(&seed);
      
      atom_sdd = bfgs_p(time, data_series, atom_i, nvalid);
      atom_i = atom_sdd.lo;//(float4) (atom_sdd.s0, atom_sdd.s1, atom_sdd.s2, atom_sdd.s3);
      
      errorf = deviance_atom(time, data_series, atom_i, nvalid);
      utility = error0 - errorf;
      //penalizes any parameter which is not a minimum
      utility -= isnan(atom_sdd.s4) * 10 + isnan(atom_sdd.s5) * 10 + isnan(atom_sdd.s6) * 10 + isnan(atom_sdd.s7) * 10;
      utility -= isinf(atom_sdd.s4) * 10 + isinf(atom_sdd.s5) * 10 + isinf(atom_sdd.s6) * 10 + isinf(atom_sdd.s7) * 10;
      //penalizes too much variability in parameters estimation
      utility -= 100 * (atom_sdd.s4 >= 1.) + 100 * (atom_sdd.s5 >= 1.) + 100 * (atom_sdd.s6 >= 1.) + 100 * (atom_sdd.s7 >= 0.1);
      barrier(CLK_LOCAL_MEM_FENCE);
      
      //put results into the  vector of utilities
      float4 best_params = select_atom(vatoms, utility, errorf, atom_sdd, atomsf, atomssd, catoms);
      //obtain the best fit from the parameters list
      cresiduals = vatoms[0].z;
      //if all the utilities are negative, break the loop
      if(vatoms[0].y < 0) break;
      barrier(CLK_LOCAL_MEM_FENCE);
      float k = 4 * (catoms + 1);
      
      switch(method)
	{
	case 1:
	  information_criterion = 2 * cresiduals + 2 * k;
	  break;
	case 2:
	  information_criterion = 2 * cresiduals + 2 * k + 2 * k * (k + 1) / (nvalid - k - 1);
	  break;
	case 3:
	  information_criterion =  2 * cresiduals + 2 * k * log((float) nvalid);
	  break;

	}
      //if the method is 3, the selection criteria is the akaike index
      //if the new AIC index increases, the loop is terminated
      /*
	If the number of parameters is > 40 nvalid (the number of atoms if > 10 nvalid)
	use AICc
	:<math>\mathrm{AICc} = \mathrm{AIC} + \frac{2k(k + 1)}{n - k - 1}</math>
	if method == 2, use BIC
	:<math> \mathrm{BIC} = {\ln(n)k - 2\ln({\hat L})}. \ </math>
	: <math>\mathrm{BIC} = n \cdot \ln(RSS/n) + k \cdot \ln(n) \ </math>
       */
      if(method > 0)
	{
	  if(information_criterion > information_criterion0)
	    return (float4) ((float) catoms, cresiduals, 0, 0);
	  information_criterion0 = information_criterion;
	}
      
      //get the residuals for the next loop
      gabor_sum_residuals_local(time, data_series, best_params, aux_vector, 1);
      error0 = cresiduals;
      ++catoms;
    }
  return (float4) ((float) catoms, cresiduals, 0, 0);
}


float4 matching_pursuit_loop(local float * time,
			     local float * data_series,
			     local float * aux_vector,
			     local float3 * vatoms,
			     local float4 atomsf[],
			     float oresiduals,
			     int oatoms,
			     int iterations,
			     ulong seed)
{
  uint i   = get_local_id(0);
  uint gid = get_group_id(0);
  //number of calculated atoms
  int catoms = 0;
  //residuals
  float cresiduals = infinity;
  while((catoms < oatoms) && (cresiduals > oresiduals))
    {
      //parameters of the best atom
      float4 best_params;
      float  innermax = 0;
      //parameters for the gabor atoms
      
      for(uint j = 0; j < iterations; ++j)
	{
	  //generate random atom
	  float4 atom_i = init_atom_p(&seed);
	  //calculate inner product between the atom and the data series
	  float2 inner = gabor_inner(time, data_series, atom_i.x, atom_i.y, atom_i.z);
	  //calculate the amplitude of each atom
	  atom_i.w = inner.x / inner.y;
	  //if the inner product is higher, then select the new atom
	  if(fabs(inner.x) > fabs(innermax))
	    {
	      innermax = inner.x;
	      best_params = atom_i;
	    }	  
	}
      //after finishing the loop, choose the best atom of the threads
      
      //thread number of each calculated atom
      vatoms[i].x = (float) i;
      //inner product of each atom
      vatoms[i].y = fabs(innermax);
      barrier(CLK_LOCAL_MEM_FENCE);
      //choose the atom which maximizes inner product
      choose_best(vatoms);
      barrier(CLK_LOCAL_MEM_FENCE);
      
      //obtain the index of the best atom
      uint best_index = (int) vatoms[0].x;
      barrier(CLK_LOCAL_MEM_FENCE);
      //copy the best parameters to the local memory
      if(i == best_index)
	atomsf[catoms] = best_params;
      barrier(CLK_LOCAL_MEM_FENCE);
      //copy the best parameters back to private memory
      best_params = atomsf[catoms];
      
      cresiduals = gabor_sum_residuals_local(time, data_series, best_params, aux_vector, 0);
      ++catoms;

    }
  return (float4) ((float) catoms, cresiduals, 0.0f, 0.0f);
}


kernel void gpu_pursuit(global float * icube,
			global float * itime,
			global float * params,
			global float * stdparams,
			global int   * natoms,
			ulong  iseed,
			int    iterations,
			int    center,
			int    minvalids,
			float  oresiduals,
			int    oatoms,
			int    algorithm,
			int    return_residuals)
{
  uint i   = get_local_id(0);
  uint gid = get_group_id(0);
  //input time series in the local memory
  local float data_series[_len_t];
  //input time data in the local memory
  local float time[_len_t];
  ulong seed = iseed + get_global_id(0);
  uint nparams = 0;
  //auxiliary vector
  local float aux_vector[_lmax_];
  local float3 vatoms[_lmax_];
  local float4 atomsf[24];
  local float4 atomssd[24];
  
  //copy data to local memory
  copy_cube_vector(icube, data_series);
  copy_cube_vector(itime, time);
  barrier(CLK_LOCAL_MEM_FENCE);

  float nvalid = count_valid(data_series, aux_vector);
  barrier(CLK_LOCAL_MEM_FENCE);
  //only perform matching pursuit in a series with more than nvalid values
  if(nvalid > minvalids)
    {
      //center series
      float mean   = 0;
      if(center)
	center_series(data_series, aux_vector, nvalid);
      float4 result;
      barrier(CLK_LOCAL_MEM_FENCE);

      //algorithm 0 is matching pursuit
      if(algorithm == 0)
	result = matching_pursuit_loop(time, data_series, aux_vector, vatoms, atomsf, oresiduals, oatoms, iterations, seed);
      //algorithm 1 is basis pursuit minimizing squared error
      //algorithm 2 is basis pursuit minimizing with akaike criterion selecting atoms
      //algorithm 3 is basis pursuit minimizing with corrected akaike criterion selecting atoms
      //algorithm 4 is basis pursuit minimizing with bayesian information criterion selecting atoms 
      else
	result = basis_pursuit_incrowd(time, data_series, aux_vector, vatoms, atomsf, atomssd, oresiduals, oatoms, seed, algorithm - 1, nvalid);
      int natomsc = (int) result.x;
            
      if(i < (int) natomsc)
	{
	  params[index_params(i, 0)] = atomsf[i].x;
	  params[index_params(i, 1)] = atomsf[i].y;
	  params[index_params(i, 2)] = atomsf[i].z;
	  params[index_params(i, 3)] = atomsf[i].w;
	  //return params error if the algorithm is one of the basis pursuit flavours
	  if(algorithm > 0)
	    {
	      stdparams[index_params(i, 0)] = atomssd[i].x;
	      stdparams[index_params(i, 1)] = atomssd[i].y;
	      stdparams[index_params(i, 2)] = atomssd[i].z;
	      stdparams[index_params(i, 3)] = atomssd[i].w;
	    }
	}
      if(i == 0) natoms[gid] = (int) natomsc;
      barrier(CLK_LOCAL_MEM_FENCE);
      //return the residuals after performing projection
      if(return_residuals)
	copy_vector_cube(icube, data_series);
    }
}

