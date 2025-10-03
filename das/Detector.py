#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HECTOR Detector Module - Semblance-based Event Detection

This module implements the Detector class which extends the DAS base class
with hyperbolic semblance-based event detection capabilities. The detector
identifies microseismic events by computing waveform coherence along geometric
hyperbolic trajectories, independent of velocity models.

Key Features:
- Model-independent detection using hyperbolic semblance scanning
- Numba-accelerated computations for performance
- Automatic event clustering and SNR-based filtering
- Visualization tools for detection results and parameter tuning

Created on Mon Oct 24 17:21:23 2022
@author: juan
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import DAS as DAS
from numba import jit, njit, prange


## COMPILED FUNCTIONS ##

@jit(nopython=True)
def sample_select(ns_window, data, dat_win, y, a, b, c, d_static):
    """
    Extract data samples along a hyperbolic trajectory (JIT-compiled).

    This function computes a hyperbola defined by parameters (a, b, c, d_static)
    and extracts a sliding window of samples from each trace along the hyperbolic
    curve. Uses Numba JIT compilation for performance.

    Parameters
    ----------
    ns_window : int
        Width of the sampling window in samples. Defines how many samples to
        extract around each hyperbola point. Should be even for symmetric windowing.
    data : numpy.ndarray
        2D array of DAS data with shape (ntrs, npts).
    dat_win : numpy.ndarray
        Pre-allocated output array with shape (ntrs, ns_window) to store
        extracted samples. Passed for memory efficiency.
    y : numpy.ndarray
        1D array of trace indices [0, 1, 2, ..., ntrs-1]. Represents spatial
        positions along the fiber.
    a : float
        Hyperbola vertex width parameter. Smaller values create wider hyperbolas.
        Controls the aperture of the hyperbolic trajectory.
    b : float
        Time offset parameter. Position of hyperbola vertex along time axis
        (in samples). Typically set to (time_sample - a) for scanning.
    c : float
        Curvature coefficient. Smaller values create higher curvature hyperbolas.
        Controls the steepness of the hyperbolic trajectory.
    d_static : float
        Lateral position parameter. Trace index of the hyperbola vertex along
        the spatial axis. Defines where the hyperbola is centered.

    Returns
    -------
    numpy.ndarray
        Data window array with shape (ntrs, ns_window) containing samples
        extracted along the hyperbola. Traces with invalid indices are zero-filled.

    Notes
    -----
    **Hyperbola equation:**
        x[k] = sqrt((1 + ((y[k] - d_static)/c)^2) * a^2) + b

    where x[k] is the time sample index for trace k.

    **Performance:**
    - JIT compilation (nopython=True) provides ~100x speedup
    - First call is slower due to compilation overhead
    - Pre-allocation of dat_win avoids memory allocation in inner loop

    **Error handling:**
    - Try/except catches index out-of-bounds errors
    - Invalid samples are zero-filled (e.g., hyperbola extends beyond data)

    Examples
    --------
    >>> ns_window = 20
    >>> dat_win = np.zeros((ntrs, ns_window))
    >>> y = np.arange(ntrs)
    >>> result = sample_select(ns_window, data, dat_win, y, a=20, b=350, c=100, d_static=500)
    """

    # Compute hyperbola: x = sqrt((1 + ((y-d)/c)^2) * a^2) + b
    x = np.sqrt( ( 1 + ( (y-d_static) /c ) **2 ) *a**2 ) + b

    # Extract samples along hyperbola for each trace
    for k in range(data.shape[0]):
        i_hyp = int(x[k])  # Time index for trace k

        try:
            # Extract symmetric window around hyperbola point
            dat_win[k,:] = data[k,i_hyp-ns_window//2:i_hyp+ns_window//2]
        except:
            # Zero-fill if hyperbola extends beyond data bounds
            dat_win[k,:] = 0.

    return dat_win



# def x_search(arr, d_vector, sem, sem2, sem_matrix, b_vector, c_vector, ns_window, dat_win, y, a, d_static, svd=False):
#     for d in np.ndenumerate(d_vector):
            
#         #print(int(np.round(100*d[1]/(d_vector.max()), decimals=1)), '%')
        
#         sem, sem2 = b_c_iter(arr, sem, sem2, b_vector, c_vector, ns_window, dat_win, y, a, d[1], svd=False)
                
#         sem2 = sem2/sem2.max() # normalized between 0 and 1 to scale sem matrix
    
    
#         if svd==False:
#             sem_matrix = np.dstack((sem_matrix, sem))
#         else:
#             sem_matrix = np.dstack((sem_matrix, sem*sem2))

#     return sem_matrix

#############################################################################

class Detector(DAS.DAS):
    """
    Semblance-based microseismic event detector for DAS data.

    This class extends the DAS base class with hyperbolic semblance scanning
    capabilities for detecting microseismic events. It implements the HECTOR
    algorithm described in Porras et al. (2024, GJI).

    The detector operates in two stages:
    1. Waveform coherence analysis using semblance along hyperbolic trajectories
    2. Event clustering and SNR-based filtering

    Inherits all DAS class methods for data loading, preprocessing, and visualization.

    Methods
    -------
    detector(ns_window, a, b_step, c_min, c_max, c_step, ...)
        Main detection engine - scans data for hyperbolic coherence
    detected_events(min_numb_detections, max_dist_detections, ...)
        Extract events from coherence time-series
    hyperbolae_tuning(a, b, d, c_min, c_max, c_step, ...)
        Visualize hyperbolic trajectories for parameter tuning
    plot_report(path_results, savefig, fig_format)
        Generate multi-panel detection report figure

    Attributes
    ----------
    sem : numpy.ndarray
        2D semblance matrix with shape (n_curvatures, n_times)
    sem_matrix : numpy.ndarray
        3D semblance volume (if lateral search used), shape (n_curv, n_times, n_positions)
    coh : numpy.ndarray
        Coherence time-series (squared sum of semblance columns)
    threshold : numpy.ndarray
        Detection threshold array (same length as coh)
    events : numpy.ndarray
        Array of detected event times (in seconds)
    curv : numpy.ndarray
        Array of curvature coefficients used in scanning
    b_step : int
        Time step used in scanning (samples)
    d_best : float
        Best lateral position (if lateral search used)
    """

    def __init__(self,file, dx, gl, fname, file_format, duration):
        """
        Initialize a Detector object by reading DAS data.

        Creates a Detector instance with all DAS class functionality plus
        semblance-based event detection capabilities.

        Parameters
        ----------
        file : str
            Full path to DAS data file (e.g., './input/data.sgy').
        dx : float
            Channel spacing in meters. Distance between consecutive DAS channels.
            Typical values: 0.5-10 m.
        gl : float
            Gauge length in meters. Physical length over which strain is averaged.
            Common values: 5-20 m.
        fname : str
            Base filename (without path/extension) for output file naming.
            Example: 'FORGE_2022_stage3'
        file_format : str
            File format identifier. Options:
            - 'segy': SEG-Y seismic format (auto-detects metadata)
            - 'tdms': National Instruments TDMS format (auto-detects metadata)
            - 'npy': NumPy binary array (requires duration parameter)
            - 'h5': HDF5 format (requires duration parameter)
        duration : float
            Total recording duration in seconds. Required for 'npy' and 'h5'
            formats. Not needed for 'segy' or 'tdms'.

        Returns
        -------
        Detector
            Detector object with DAS data loaded and ready for processing.

        Examples
        --------
        Create detector from SEG-Y file:
        >>> det = Detector('./input/forge.sgy', dx=1.02, gl=10.0,
        ...                fname='forge', file_format='segy', duration=None)

        Create detector from NumPy array:
        >>> det = Detector('./input/synthetic.npy', dx=2.0, gl=10.0,
        ...                fname='synth', file_format='npy', duration=60.0)

        See Also
        --------
        DAS.__init__ : Base class constructor documentation
        """
        DAS.DAS.__init__(self,file, dx, gl, fname, file_format, duration)
        
    
    def __svd(self, arr):
        """
        Compute SVD-based coherence metric (private method, currently unused).

        Calculates eigenvalue-based coherence using singular value decomposition
        of the data covariance matrix. Can be used as alternative to semblance
        for weighting detection strength.

        Parameters
        ----------
        arr : numpy.ndarray
            Data window with shape (ntrs, ns_window).

        Returns
        -------
        float
            Normalized coherence metric based on dominant eigenvalue.
            Returns 0 if array is all-zero or denominator is zero.

        Notes
        -----
        - Currently NOT used in main detection workflow (svd=False by default)
        - Could not be compiled with Numba (no performance gain)
        - Returns s_n = (s[0] - mean(s[1:])) / mean(s[1:])
        - where s are sorted singular values of covariance matrix
        """
        if arr.all()==False:
            return 0.
        else:
            u, s, v = np.linalg.svd(np.cov(arr))
            m = s.shape[0]  # number of traces

            op = np.sum(s[1:]/(m-1))  # Mean of non-dominant eigenvalues
            s_n = (s[0] - op) / op     # Normalized signal strength

            return 0. if op==0. else s_n

    def __semblance_func(self, arr):
        """
        Compute semblance (waveform coherence) for a data window (private method).

        The semblance function measures the coherence of waveforms across multiple
        traces. It ranges from 0 (no coherence) to 1 (perfect coherence).

        Parameters
        ----------
        arr : numpy.ndarray
            Data window with shape (ntrs, ns_window) where:
            - ntrs: Number of traces
            - ns_window: Number of samples in window

        Returns
        -------
        float
            Semblance value in range [0, 1]:
            - ~0: Random noise, no coherent signal
            - ~1: Perfect coherence across all traces
            Returns 1e-16 if denominator is zero (avoid division by zero).

        Notes
        -----
        **Mathematical definition:**
            Semblance = [sum(sum(arr, axis=0)^2)] / [ntr * sum(arr^2)]

        where:
        - Numerator: Energy of stacked traces (coherent energy)
        - Denominator: Total energy across all traces

        **Interpretation:**
        - High semblance indicates coherent wavefront (likely seismic event)
        - Low semblance indicates incoherent noise
        - Independent of amplitude (normalized by total energy)

        **Performance:**
        - Operates on small windows (typically 20 samples)
        - Called millions of times during scanning (inner loop)
        - Vectorized numpy operations for efficiency

        Examples
        --------
        >>> window = data[:, 100:120]  # 20-sample window
        >>> coherence = self.__semblance_func(window)
        >>> # coherence ~0.8 suggests coherent signal
        """
        ntr = arr.shape[0]
        num = np.sum(np.square(np.sum(arr, axis=0)))  # Stacked energy
        den = np.sum(arr**2) * ntr                     # Total energy
        return 1e-16 if den==0. else num/den
    
   
    
    def hyperbolae_tuning(self, a, b, d, c_min, c_max, c_step, path_results='./report/', savefig=False, fig_format='png'):
        """
        Visualize hyperbolic scanning trajectories overlaid on DAS data.

        This method plots a family of hyperbolic curves on top of your DAS data
        to help you verify that your parameter choices adequately cover the events
        of interest. Essential for parameter tuning before running the detector.

        Parameters
        ----------
        a : float
            Hyperbola vertex width parameter. Smaller values create wider hyperbolas.
            Controls the aperture of the hyperbolic scanning pattern.
            Typical range: 10-50.
        b : float
            Time position of hyperbola vertex along the sample axis (sample index).
            Defines where the hyperbola apex is positioned in time.
            Example: 375 for scanning around sample 375.
        d : float
            Spatial position of hyperbola vertex along the trace axis (trace index).
            Defines where the hyperbola is centered spatially.
            Example: 892 for centering at trace 892.
        c_min : float
            Minimum curvature coefficient. Smaller values = higher curvature
            (tighter hyperbola). Start of curvature range to visualize.
            Typical: 50-100.
        c_max : float
            Maximum curvature coefficient. Larger values = lower curvature
            (flatter hyperbola). End of curvature range to visualize.
            Typical: 200-500.
        c_step : float
            Step size between c_min and c_max. Controls density of plotted
            hyperbolas. Smaller step = more hyperbolas displayed.
            Can be fractional (e.g., 2.5) for finer visualization.
            Typical: 5-10.
        path_results : str, optional
            Directory path for saving output figure. Must exist before calling.
            Default: './report/'
        savefig : bool, optional
            If True, saves figure to disk. If False, displays interactively.
            Default: False.
        fig_format : str, optional
            Output figure format. Options: 'png', 'pdf', 'jpg', 'svg'.
            Default: 'png'.

        Returns
        -------
        None
            Displays matplotlib figure showing DAS data with overlaid hyperbolic
            trajectories. Saves to file if savefig=True.

        Notes
        -----
        **Usage workflow:**
        1. Run this method BEFORE calling detector() to verify parameters
        2. Adjust c_min, c_max, a to ensure hyperbolas match event moveout
        3. Check that hyperbolas bracket the event of interest
        4. If hyperbolas don't fit, adjust parameters and re-run

        **Parameter relationships:**
        - Smaller `a` → wider vertex → covers more traces laterally
        - Smaller `c` → tighter curve → matches near-field events
        - Larger `c` → flatter curve → matches far-field events
        - `d` should be near the center of the event spatially

        **Output file:**
        - Filename: {path_results}{fname}_hyperbola_tuning.{fig_format}
        - Example: './report/FORGE_data_hyperbola_tuning.png'

        Examples
        --------
        Visualize parameter space for FORGE-like data:
        >>> det.hyperbolae_tuning(
        ...     a=20, b=375, d=892,
        ...     c_min=70, c_max=500, c_step=5,
        ...     savefig=True
        ... )

        Quick interactive check:
        >>> det.hyperbolae_tuning(a=20, b=200, d=500,
        ...                       c_min=50, c_max=300, c_step=10)

        See Also
        --------
        detector : Main detection method that uses these parameters
        """
                
        fontsize = 10
        plt.rcParams['axes.linewidth'] = 0.3
        plt.rc('xtick', labelsize=8) 
        plt.rc('ytick', labelsize=8)
        
        fig, ax = plt.subplots(figsize=(10,5))
        
        ax.imshow(self.traces, cmap='seismic', aspect='auto')
        ax.set_ylabel('Number of traces', fontsize=fontsize)
        ax.set_xlabel('Samples', fontsize=fontsize)
       
        y = np.arange(self.ntrs)
        c_vector = np.arange(c_min, c_max, c_step) # range of curvatures. The smaller the number --> the higher the curvature is.
        b = b-a
        
                
        for c in c_vector:
            x = np.sqrt( ( 1 + ( (y-d) /c ) **2 ) *a**2 ) + b
            ax.plot(x, y, c='gray', alpha=.5)

        fig.tight_layout()
        
        if savefig is True:
            plt.savefig(path_results + self.fname + '_hyperbola_tuning.' + fig_format, dpi=300)
                        
    
    def b_c_iter(self, sem, sem2, b_vector, c_vector, ns_window, dat_win, y, a, d_static, svd=False):
        print('Scanning waveform coherence ....')
        for b in np.ndenumerate(b_vector):
            #print(int(np.round(100*(b[1]*self.dt)/(self.npts*self.dt), decimals=0)), '%')
            
            # selection of samples
            for c in np.ndenumerate(c_vector):
                
                # selection of samples
                dat_win = sample_select(ns_window, self.traces, dat_win, y, a, b[1], c[1], d_static)
                sem[c[0],b[0]] = self.__semblance_func(dat_win)
                sem2[c[0],b[0]] = self.__svd(dat_win) if svd==True else None
        
        return sem, sem2
    
    
    def x_search(self, d_vector, sem, sem2, sem_matrix, b_vector, c_vector, ns_window, dat_win, y, a, d_static, svd=False):
        
        for d in np.ndenumerate(d_vector):
                
            print(int(np.round(100*d[1]/(d_vector.max()), decimals=1)), '%')
            
            sem, sem2 = self.b_c_iter(sem, sem2, b_vector, c_vector, ns_window, dat_win, y, a, d[1], svd)
                    
            sem2 = sem2/sem2.max() # normalized between 0 and 1 to scale sem matrix
        
        
            if svd==False:
                sem_matrix = np.dstack((sem_matrix, sem))
            else:
                sem_matrix = np.dstack((sem_matrix, sem*sem2))

        return sem_matrix
    
     
    def detector(self,
                  ns_window,
                  a,
                  b_step,
                  c_min, c_max, c_step,
                  d_static=None, d_min=None, d_max=None, d_step=None,
                  shift=0, svd=False, lat_search=False):
        
        
        
        int(ns_window)+1 if int(ns_window)%2 != 0 else int(ns_window)
        
        y = np.arange(self.ntrs)
        
        self.b_step = b_step
        c_vector = np.arange(c_min, c_max+c_step, c_step)
        self.curv = c_vector
        b_vector = np.arange(0, self.npts, b_step) - a
    
        dat_win = np.zeros((self.ntrs, ns_window))
                        
        sem = np.zeros((len(c_vector), len(b_vector)))
        sem2 = sem.copy()
        sem_matrix = np.zeros((len(c_vector), len(b_vector)))
        
               
        ##############----- PERFORM SCANNING -------------#########
        
        if lat_search == False and d_static is not None:
            sem, sem2 = self.b_c_iter(sem, sem2, b_vector, c_vector, ns_window, dat_win, y, a, d_static, svd=False)
                        
            sem2 = sem2/sem2.max() # svd weight normalized between 0 and 1 to scale sem matrix
            sem = sem*sem2 if svd==True else sem
            
        elif lat_search == False and d_static is None:
            raise Exception('Must enter a value for d_static')
        
        
        elif lat_search == True:
            
            if d_min == None and d_max == None and d_step == None:
                d_min, d_max, d_step = 0, int(self.ntrs * self.dx), int(self.ntrs/10)
            
            d_vector = np.arange(d_min, d_max+d_step, d_step)
            
            #args = [d_vector, sem, sem2, sem_matrix, b_vector, c_vector, ns_window, dat_win, y, a, d_static, svd]
            
            #import multiprocessing
            #with multiprocessing.Pool(processes=4) as pool:
            #    sem_matrix = pool.starmap(self.x_search, args)
            
            # import multiprocessing
            # with multiprocessing.Pool(processes=4) as pool:
            #     sem_matrix = pool.apply_async(self.x_search,
            #                                   args=(d_vector,
            #                                         sem,
            #                                         sem2,
            #                                         sem_matrix,
            #                                         b_vector,
            #                                         c_vector,
            #                                         ns_window,
            #                                         dat_win,
            #                                         y,
            #                                         a,
            #                                         d_static,
            #                                         svd))
            
            # parallelize this function
            sem_matrix = self.x_search(d_vector, sem, sem2, sem_matrix, b_vector, c_vector, ns_window, dat_win, y, a, d_static, svd)
            
                    
            sem_matrix = sem_matrix[:,:,1:]
            d_position = np.where(sem_matrix == np.max(sem_matrix))[2]
            self.d_best = d_vector[d_position[0]]
            sem = sem_matrix[:,:,d_position]
                    
        self.sem_matrix = sem_matrix
        self.sem = sem
        
        #####################################################
        

    # def detector(self,
    #               ns_window,
    #               a,
    #               b_step,
    #               c_min, c_max, c_step,
    #               d_static=None, d_min=None, d_max=None, d_step=None,
    #               shift=0, svd=False, lat_search=False):
        
        
        
    #     int(ns_window)+1 if int(ns_window)%2 != 0 else int(ns_window)
        
    #     y = np.arange(self.ntrs)
    #     self.b_step = b_step
    #     c_vector = np.arange(c_min, c_max+c_step, c_step)
    #     self.curv = c_vector
    #     b_vector = np.arange(0, self.npts, b_step) - a
    
    #     dat_win = np.zeros((self.ntrs, ns_window))
       
    #     sem = np.zeros((len(c_vector), len(b_vector)))
    #     sem2 = sem.copy()
    #     sem_matrix = np.zeros((len(c_vector), len(b_vector)))
        
    #     ##############----- PERFORM SCANNING -------------#########
        
    #     if lat_search == False and d_static is not None:
            
    #         for b in np.ndenumerate(b_vector):
    #             print(int(np.round(100*(b[1]*self.dt)/(self.npts*self.dt), decimals=0)), '%')
                
    #             # selection of samples
    #             for c in np.ndenumerate(c_vector):
                    
    #                 # selection of samples
    #                 dat_win = sample_select(ns_window, self.traces, dat_win, y, a, b[1], c[1], d_static)
    #                 sem[c[0],b[0]] = semblance_func(dat_win)
    #                 sem2[c[0],b[0]] = self.__svd(dat_win) if svd==True else None
                                        
            
    #         sem2 = sem2/sem2.max() # svd weight normalized between 0 and 1 to scale sem matrix
    #         sem = sem*sem2 if svd==True else sem
            
    #     elif lat_search == False and d_static is None:
    #         raise Exception('Must enter a value for d_static')
        
        
    #     elif lat_search == True:
            
    #         if d_min == None and d_max == None and d_step == None:
    #             d_min, d_max, d_step = 0, int(self.ntrs * self.dx), int(self.ntrs/10)
            
    #         d_vector = np.arange(d_min, d_max+d_step, d_step)
            
    #         for d in np.ndenumerate(d_vector):
    #             print(int(np.round(100*d[1]/(d_vector.max()), decimals=1)), '%')
                
    #             for b in np.ndenumerate(b_vector):
                    
    #                 for c in np.ndenumerate(c_vector):
                        
    #                     dat_win = sample_select(ns_window, self.traces, dat_win, y, a, b[1], c[1], d[1])
    #                     sem[c[0],b[0]] = semblance_func(dat_win)
    #                     sem2[c[0],b[0]] = self.__svd(dat_win) if svd==True else None
                        
                        
    #             sem2 = sem2/sem2.max() # normalized between 0 and 1 to scale sem matrix

            
    #             if svd==False:
    #                 sem_matrix = np.dstack((sem_matrix, sem))
    #             else:
    #                 sem_matrix = np.dstack((sem_matrix, sem*sem2))
                    
    #         sem_matrix = sem_matrix[:,:,1:]
    #         d_position = np.where(sem_matrix == np.max(sem_matrix))[2]
    #         self.d_best = d_vector[d_position[0]]
    #         sem = sem_matrix[:,:,d_position]
                    
    #     self.sem_matrix = sem_matrix
    #     self.sem = sem
        
    #     #####################################################

    def plot_report(self, path_results='./report/', savefig=False, fig_format='png'):
        '''
        

        Parameters
        ----------
        path_results : String, optional
            Path where to store the results. The default is './report/'.
        savefig : Boolean, optional
            Set it to True if you want to save the figure. The default is False.
        fig_format : String, optional
            Format in which to save the figure. The default is png.

        Returns
        -------
        A matplotlib plot with the results of the detector

        '''
        
        width = 15/2.54
        fontsize = 10
        plt.rcParams['axes.linewidth'] = 0.3
        plt.rc('xtick', labelsize=8) 
        plt.rc('ytick', labelsize=8)
        
        fig, ((ax1, ax2, ax3)) = plt.subplots(3, 1,  figsize=(width,4), sharex=True, constrained_layout=False, gridspec_kw={'height_ratios': [1, 2, 3]})
        
        cut = 3 # remove the last 3 columns of the semblance which are not properly computed for every scanning

        time = np.arange(self.npts-(self.b_step*cut))*self.dt
        coh_vector, threshold = self.coh, self.threshold 
        
        time_tmp = np.linspace(time.min(), time.max(), len(coh_vector))
        
        ax1.plot(time_tmp, coh_vector, linewidth=.4)
        ax1.plot(time_tmp, threshold, 'r--', linewidth=.4)
        ax1.set_ylabel('Max. coherence', fontsize=fontsize)
        ax1.spines[['right', 'top']].set_visible(False)
                
        events_clean=self.events
        
        for ev in events_clean:
            ax1.axvline(x = ev, ls='--', color='k',lw=.6)

        divider = make_axes_locatable(ax1)
        cax2 = divider.append_axes("right", size="1.4%", pad=.05)
        cax2.axis('off')
        
        ## AX2 ##
        depth=np.arange(self.ntrs)*self.dx
        im2 = ax2.imshow(self.sem, extent=[min(time),max(time),max(self.curv),min(self.curv)],cmap='viridis',aspect='auto')
        ax2.set_ylabel('Curv. coeff.', fontsize=fontsize)
        
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='1.4%', pad=.05)
        fig.colorbar(im2, cax=cax, orientation='vertical')
        cax.xaxis.set_ticks_position("none")
                
        ## AX3 ##
        
        im3 = ax3.imshow(self.traces[:,:-cut*self.b_step],
                         extent=[min(time), max(time), max(depth), min(depth)],
                         cmap='seismic',
                         aspect='auto')
        
        ax3.set_ylabel('Linear Fiber Length [m]', fontsize=fontsize)
        ax3.set_xlabel('Relative time [s]', fontsize=fontsize)
        
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', size='1.4%', pad=.05)
        fig.colorbar(im3, cax=cax, orientation='vertical')
        cax.xaxis.set_ticks_position("none")

        fig.tight_layout()
  
        if savefig is True:
            plt.savefig(path_results + self.fname + '_detection_plot.' + fig_format, dpi=300)
                        
            del fig, ax1, ax2, ax3, coh_vector, threshold, events_clean
        

############################
    def detected_events(self, min_numb_detections, max_dist_detections, path_results='./report/'):

        import csv
        from scipy import stats
        from scipy import signal
        path_results=path_results
        from obspy import UTCDateTime

        cut = 3 # remove the last 3 columns of the semblance which are not properly computed for every scanning
        self.sem = self.sem[:,:-cut]
        
        coh_vector = np.sum(np.square(self.sem), axis=0) # GIAN MARIA
        #coh_vector = np.sum(self.sem, axis=0) #JUAN
        
        
        
        #define a noise threshold above which to make detections 
        
        noise=stats.trim_mean(coh_vector, 0.05)

        threshold = np.full_like(coh_vector, noise) 

        self.coh, self.threshold = coh_vector, threshold

        duration = self.npts*self.dt

        tax = np.linspace(0, duration, len(self.coh))

        pos = np.where(self.coh > self.threshold)
        
        #y_logs=[]
        #for k, l in enumerate(coh_vector):
        #    if k < len(coh_vector)-1:
        #        y_logs.append(np.log(coh_vector[k+1]/coh_vector[k]))
        #
        #x_logs=np.linspace(1,len(y_logs),len(y_logs))
        #y_logs=np.cumsum(y_logs)
        #plt.plot(x_logs,y_logs)
        #noise1=np.sqrt(np.mean(np.square(y_logs)))
        #plt.axhline(y = noise1, color = 'r')
        #plt.savefig(fname+'.png', dpi=150)
        

        diff = []
        for i in pos:
            d = np.abs(self.coh[i] - self.threshold[i])
            diff.append(d)
        diff = np.array(diff)
        
        #print(self.coh[pos])

        #group close detections
        def grouper(iterable):
            prev = None
            group = []
            for item in iterable:
                if prev is None or item - prev <= max_dist_detections:
                    group.append(item)
                else:
                    yield group
                    group = [item]
                prev = item
            if group:
                yield group
        
        pos1=(dict(enumerate(grouper(pos[0]), 1)))
        #print(pos1)
        
        pos_ev=[]
        for k,i in pos1.items():
            
            ## check that in each "event" there are at least a certain number of detections 
            semb_coeff=(self.coh[i])
            
            if len(i)>=min_numb_detections:# and np.mean(semb_coeff)/noise>=1.8:
                detect_= np.min(i)
                #if detect_<=2: #here I want to remove continuations of events from the previous file
                #    continue
                noise_det=np.sqrt(np.mean(np.square(coh_vector[detect_-22:detect_-2])))
                if detect_ <=22:
                    noise_det=noise*2.0

                signal_det=np.sqrt(np.mean(np.square(semb_coeff[:30])))
                #signal_det=np.sqrt(np.mean(np.square(coh_vector[detect_:detect_+30])))
                SNR_det=10*np.log(signal_det/noise_det)
                #print(SNR_det,'SNR',detect_)
                if SNR_det>=6:
                    pos_ev.append(np.min(i))
                
                if len (i) > 60:
                    #a=(signal.find_peaks(semb_coeff,height=0.002))[0][1:]
                    #g=c['peak_heights']
                    sample_diff=np.diff(semb_coeff,n=10)[15:]
                    max_sample_diff=np.amax(sample_diff)
                    detection=int(np.where(sample_diff==max_sample_diff)[0])+15
                    #print(sample_diff,max_sample_diff)
                    #a=signal.argrelmax(semb_coeff,order=30)[0][1:]
                    noise_=np.sqrt(np.mean(np.square(semb_coeff[detection-17:detection-2])))
                    signal_=np.sqrt(np.mean(np.square(semb_coeff[detection:detection+20])))
                    SNR=10*np.log(signal_/noise_)
                    #print(SNR,i[int(detection)])
                    if  SNR>4:# and a[0][0]:
                        pos_ev.append((i[int(detection)]))
        
        events = tax[pos_ev]
        events = np.around(events, decimals=3)
        fname2 = [self.fname] * len(events)
        
               
        abs_time = []
        
        if self.format != 'tdms':
            for i in events:
                abs_time.append(self.starttime + i)
        else:
            for time_ev in events:
                time_ev=("%.3f" % float(time_ev))
                secs, ms = str(time_ev).split('.')
                secs, ms = int(secs), int(ms)
                tmp= self.starttime + np.timedelta64(secs,'s') + np.timedelta64(ms,'ms')
                abs_time.append(tmp)
        
               
        events2 = events.tolist()
        if np.any(events)==False:
            None
        else:
            with open('{}{}.out'.format(path_results, self.fname), 'w') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerows(zip(fname2, events2, abs_time))
                                
        # ADD HEADERS
        self.events = events
        return self.events, self.coh, self.threshold
        del fname2, events2, abs_time
    
  