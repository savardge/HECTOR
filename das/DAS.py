#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DAS Data Processing Module for HECTOR

This module provides the base DAS class for handling Distributed Acoustic Sensing data.
It supports multiple file formats and provides preprocessing capabilities including
filtering, downsampling, FK filtering, and trace normalization.

Created on Mon Oct 24 17:21:23 2022
@author: juan
"""

import numpy as np
import matplotlib.pyplot as plt

def readerh5(file):
    """
    Read DAS data from HDF5 format files.

    This function extracts strain data from HDF5 files with a specific structure
    containing receiver IDs and strain tensors.

    Parameters
    ----------
    file : str
        Path to the HDF5 file containing DAS data.

    Returns
    -------
    numpy.ndarray
        2D array of strain data with shape (ntrs, npts) where:
        - ntrs: Number of DAS traces (spatial channels)
        - npts: Number of time samples

    Notes
    -----
    The function expects HDF5 files with the following structure:
    - 'receiver_ids_ELASTIC_point': Receiver ID array
    - 'point/strain': Strain tensor data (extracts z-component, index 2)
    """
    import h5py
    data = h5py.File(file,'r')
    ids = data['receiver_ids_ELASTIC_point']
    rec = ids[:]
    if 'strain' in data['point'].keys():
        ezz = data['point']['strain'][:,2,:]  # Extract z-component of strain
        str_ezz = np.zeros_like(ezz)
        for i in range(len(ezz)):
            tmp = np.where(rec == i)
            str_ezz[i,:] = ezz[tmp,:]
            strain = str_ezz.T
    return strain.T

#%%
class DAS:
    """
    Base class for Distributed Acoustic Sensing (DAS) data handling.

    This class provides fundamental DAS data processing capabilities including:
    - Reading multiple file formats (SEG-Y, TDMS, NumPy, HDF5)
    - Data preprocessing (filtering, downsampling, normalization)
    - FK domain filtering for coherent noise removal
    - Visualization tools for time-domain and FK-domain analysis
    - STA/LTA characteristic function computation

    Attributes
    ----------
    traces : numpy.ndarray
        2D array of DAS data with shape (ntrs, npts)
    ntrs : int
        Number of traces (spatial channels along fiber)
    npts : int
        Number of time samples per trace
    dt : float
        Time sampling interval (seconds)
    dx : float
        Channel spacing (meters)
    gl : float
        Gauge length (meters) - sensing length of each DAS channel
    sampling_rate : float
        Sampling frequency (Hz)
    starttime : obspy.UTCDateTime or numpy.datetime64
        Absolute start time of the recording
    fname : str
        File name (used for output file naming)
    format : str
        Input file format ('segy', 'tdms', 'npy', or 'h5')
    """

    def __init__(self, file, dx, gl, fname, file_format, duration=None):
        """
        Initialize a DAS object by reading data from file.

        This constructor reads DAS data from various file formats and extracts
        metadata including sampling rate, start time, and array geometry.

        Parameters
        ----------
        file : str
            Full path to the DAS data file (e.g., './input/my_data.sgy').
        dx : float
            Channel spacing in meters. This is the distance between consecutive
            DAS sensing elements along the fiber. Typical values: 0.5-10 m.
        gl : float
            Gauge length in meters. This is the physical length over which strain
            is averaged for each DAS channel. Common values: 5-20 m.
        fname : str
            Base filename (without path or extension) used for naming output files.
            Example: 'FORGE_data_2022'
        file_format : str
            File format identifier. Must be one of:
            - 'segy': SEG-Y seismic format (auto-detects metadata)
            - 'tdms': National Instruments TDMS format (auto-detects metadata)
            - 'npy': NumPy binary array (requires duration parameter)
            - 'h5': HDF5 format (requires duration parameter)
        duration : float, optional
            Total duration of the recording in seconds. Required for 'npy' and 'h5'
            formats where temporal metadata is not embedded. Not needed for 'segy'
            or 'tdms' formats. Default: None.

        Raises
        ------
        Exception
            If duration is not specified for 'h5' or 'npy' format files.

        Examples
        --------
        Read SEG-Y file (auto-detects duration):
        >>> das = DAS('./input/data.sgy', dx=1.02, gl=10.0,
        ...           fname='data', file_format='segy')

        Read NumPy file (requires duration):
        >>> das = DAS('./input/data.npy', dx=2.0, gl=10.0,
        ...           fname='data', file_format='npy', duration=60.0)

        Notes
        -----
        - SEG-Y and TDMS formats automatically extract sampling rate and start time
        - NumPy and HDF5 formats assign current time as start time (placeholder)
        - Data array shape is always (ntrs, npts) = (channels, time_samples)
        - For optimal performance, ensure input files are not corrupted
        """
       
        print('Reading : ' + file)
        
        if file_format =='tdms':
            from nptdms import TdmsFile
            tdms_file = TdmsFile(file)
            self.sampling_rate = tdms_file.properties['SamplingFrequency[Hz]']
            self.starttime = tdms_file.properties['CPUTimeStamp']
            self.gl=tdms_file.properties['GaugeLength']
            traces = (tdms_file.as_dataframe().to_numpy()).T
            traces=traces

            self.ntrs, self.npts = traces.shape
            self.dt=1./self.sampling_rate
            self.fname = fname
            self.sampling_rate = tdms_file.properties['SamplingFrequency[Hz]']
            del tdms_file

        elif file_format =='segy':
            from obspy.io.segy.core import  _read_segy
            das_data = _read_segy(file,format='segy',unpack_trace_headers=True)
            traces = np.stack([trace.data for trace in das_data])
                                                
            self.ntrs, self.npts = traces.shape
            self.dt = das_data[0].stats.delta
            self.fname = fname
            self.starttime = das_data[0].stats.starttime
            self.sampling_rate = das_data[0].stats.sampling_rate
            del das_data
                        
        elif file_format =='h5':
            from obspy import UTCDateTime
            traces = readerh5(file)
            self.ntrs, self.npts = traces.shape
            
            if duration is None:
                raise Exception('the duration parameter has to be specified in the case of h5 type files')
                      
            self.dt = duration / self.npts
            self.sampling_rate = 1/self.dt
            self.fname = fname
            self.starttime = UTCDateTime.now()
        
        elif file_format =='npy':
            from obspy import UTCDateTime
            traces = np.load(file)
            self.ntrs, self.npts = traces.shape
            
            if duration is None:
                raise Exception('the duration parameter has to be specified in the case of npy type files')

            self.dt = duration / self.npts
            self.sampling_rate = 1/self.dt
            self.tax = np.arange(0, traces.shape[0], self.dt)
            
            self.fname = fname
            self.traces = traces
            self.starttime = UTCDateTime.now()
                        
        else:                
            print("Only tdms, sgy, npy and h5 file formats are supported")
                    
        self.traces = traces
        self.dx = dx
        self.gl = gl
        self.format = file_format

        
    def __downsample(self, data, sampling_rate):
        """
        Downsample DAS data to a new sampling rate (private method).

        Uses Fourier-based resampling with a Hann window for anti-aliasing.
        Updates object attributes (sampling_rate, dt, npts, traces) in-place.

        Parameters
        ----------
        data : numpy.ndarray
            Input data array with shape (ntrs, npts).
        sampling_rate : float
            Target sampling rate in Hz. Should be lower than current sampling_rate
            to avoid aliasing. Recommended: <= 0.5 * current_sampling_rate.

        Returns
        -------
        numpy.ndarray
            Downsampled data with shape (ntrs, new_npts).

        Notes
        -----
        - Always apply low-pass filter before downsampling to prevent aliasing
        - The method updates self.sampling_rate, self.dt, self.npts in-place
        - Uses scipy.signal.resample with Hann window for smooth frequency response

        Examples
        --------
        Downsample from 4000 Hz to 500 Hz:
        >>> data_resampled = self.__downsample(data, 500)
        """
        from scipy.signal import resample

        sampling_rate = int(sampling_rate)
        new = int(sampling_rate*self.npts/self.sampling_rate)
        data2 = resample(data, new, window='hann', axis=1)
        self.traces = data2
        self.ntrs, self.npts = data2.shape
        self.sampling_rate = sampling_rate
        self.dt = 1/self.sampling_rate
        return data2


    def __filter(self, data, ftype, fmin, fmax, order=4):
        """
        Apply Butterworth frequency filter to DAS data (private method).

        Implements zero-phase filtering using second-order sections (SOS)
        for numerical stability. Supports bandpass, highpass, and lowpass filters.

        Parameters
        ----------
        data : numpy.ndarray
            Input data array with shape (ntrs, npts).
        ftype : str
            Filter type. Must be one of:
            - 'bandpass': Retains frequencies between fmin and fmax
            - 'highpass': Retains frequencies above fmin (fmax ignored)
            - 'lowpass': Retains frequencies below fmax (fmin ignored)
        fmin : float
            Minimum frequency cutoff in Hz. Used for 'bandpass' and 'highpass'.
            Should be > 0 and < Nyquist frequency (0.5 * sampling_rate).
        fmax : float
            Maximum frequency cutoff in Hz. Used for 'bandpass' and 'lowpass'.
            Should be < Nyquist frequency and > fmin for bandpass.
        order : int, optional
            Butterworth filter order. Higher order = steeper roll-off but may
            introduce ringing artifacts. Typical values: 2-8. Default: 4.

        Returns
        -------
        numpy.ndarray
            Filtered data with same shape as input (ntrs, npts).

        Notes
        -----
        - Uses second-order sections (SOS) for improved numerical stability
        - Updates self.traces and shape attributes in-place
        - Apply low-pass filter before downsampling to prevent aliasing
        - For bandpass: ensure fmin < fmax < 0.5*sampling_rate

        Examples
        --------
        Bandpass filter between 10-250 Hz:
        >>> filtered = self.__filter(data, 'bandpass', 10, 250, order=4)
        """
        from scipy.signal import butter, sosfilt

        if ftype =='bandpass':
            sos = butter(order, [fmin,fmax], 'bandpass', fs=self.sampling_rate, output='sos')

        elif ftype =='highpass':
            sos = butter(order, fmin, 'highpass', fs=self.sampling_rate, output='sos')

        elif ftype =='lowpass':
            sos = butter(order, fmax, 'lowpass', fs=self.sampling_rate, output='sos')

        data2 = sosfilt(sos, data, axis=1)

        self.traces = data2
        self.ntrs, self.npts = data2.shape
        return data2
        

    def denoise(self, data, ftype=None, fmin=None, fmax=None, sampling_rate_new=None, k0=False, low_vel_events=False, order=4):
        """
        Apply comprehensive denoising workflow to DAS data.

        This method applies a sequence of preprocessing operations:
        1. Downsampling (if sampling_rate_new is specified)
        2. Frequency filtering (if ftype is specified)
        3. Trace normalization
        4. FK filtering (if k0 or low_vel_events is True)

        All operations are performed IN-PLACE, modifying self.traces directly.

        Parameters
        ----------
        data : numpy.ndarray
            Input DAS data array, typically self.traces with shape (ntrs, npts).
        ftype : str, optional
            Type of frequency filter to apply. Options:
            - 'bandpass': Bandpass filter (requires both fmin and fmax)
            - 'highpass': Highpass filter (requires fmin only)
            - 'lowpass': Lowpass filter (requires fmax only)
            - None: No frequency filtering applied (default)
        fmin : float, optional
            Minimum frequency cutoff in Hz. Required for 'bandpass' and 'highpass'.
            Example: 1 Hz for removing very low frequencies. Default: None.
        fmax : float, optional
            Maximum frequency cutoff in Hz. Required for 'bandpass' and 'lowpass'.
            Must be < Nyquist frequency (0.5 * sampling_rate).
            Example: 249 Hz for 500 Hz sampling rate. Default: None.
        sampling_rate_new : float, optional
            Target sampling rate in Hz for downsampling. If specified, data is
            first low-pass filtered and then resampled. Should be significantly
            lower than original rate (e.g., 4000 Hz → 500 Hz). Default: None.
        k0 : bool, optional
            If True, attenuates coherent linear noise parallel to fiber axis
            (mapped to k=0 wavenumber in FK domain). Useful for removing fiber-
            parallel surface waves or cable noise. Default: False.
        low_vel_events : bool, optional
            If True, attenuates low-velocity hyperbolic events in FK domain.
            Useful for removing slow-moving coherent noise (e.g., ground roll).
            Default: False.
        order : int, optional
            Order of Butterworth filter. Higher values give steeper roll-off
            but may cause ringing. Typical range: 2-8. Default: 4.

        Returns
        -------
        None
            Method modifies self.traces in-place. Updated data accessible via
            self.traces attribute.

        Warnings
        --------
        - This operation OVERWRITES self.traces - save original data if needed
        - Always apply low-pass filter when downsampling to prevent aliasing
        - FK filtering parameters are tuned for FORGE dataset (see __fk_filt docs)

        Examples
        --------
        Complete denoising workflow (FORGE-style):
        >>> das.denoise(
        ...     das.traces,
        ...     sampling_rate_new=500,
        ...     ftype='bandpass',
        ...     fmin=1,
        ...     fmax=249,
        ...     k0=True,
        ...     low_vel_events=True
        ... )

        Simple frequency filtering only:
        >>> das.denoise(das.traces, ftype='bandpass', fmin=10, fmax=100)

        Notes
        -----
        Processing order is critical:
        1. Downsample first (reduces computational cost for subsequent steps)
        2. Apply frequency filter on downsampled data
        3. Normalize traces (removes amplitude variations)
        4. Apply FK filter last (works on normalized, filtered data)
        """
        
        if sampling_rate_new is not None:
            
            traces = self.__filter(data, 'lowpass', fmin, fmax) # lowpass before downsampling
            traces = self.__downsample(traces, sampling_rate_new) # downsampling of the data
        
        else:
            
            traces = data
        
        if ftype is not None:
            
            traces = traces - traces.mean() # detrend the data
            traces = self.__filter(traces, ftype, fmin, fmax) # Bandpass filtering the data
        
        else:
            
            traces = data
        
        traces = self.__trace_normalization(traces) # normalize the data
        
        if k0 or low_vel_events:
            traces = self.__fk_filt(traces, k0, low_vel_events) #FK filtering the data
        
        self.traces = traces
        self.ntrs,self.npts = traces.shape



        
    def data_select(self, starttime=None, endtime=None, startlength=None, endlength=None):
        """
        Extract a spatiotemporal subset of DAS data.

        This method selects a rectangular window from the DAS data array, allowing
        you to focus on specific time intervals and spatial segments along the fiber.
        Operation is performed IN-PLACE, modifying self.traces.

        Parameters
        ----------
        starttime : float, optional
            Start time in seconds relative to recording start. If None, uses
            beginning of recording (0 seconds). Must be >= 0. Default: None.
        endtime : float, optional
            End time in seconds relative to recording start. If None or -1, uses
            end of recording. Must be > starttime. Default: None.
        startlength : float, optional
            Start position along fiber in meters. If None, uses beginning of
            fiber (0 m). Must be >= 0 and < total fiber length. Default: None.
        endlength : float, optional
            End position along fiber in meters. If None or -1, uses end of fiber.
            Must be > startlength. Default: None.

        Returns
        -------
        None
            Method modifies self.traces, self.ntrs, self.npts in-place.

        Warnings
        --------
        - This operation OVERWRITES self.traces - save original if needed
        - Time parameters are in seconds (not sample indices)
        - Length parameters are in meters (not channel indices)

        Examples
        --------
        Select first 5 seconds:
        >>> das.data_select(starttime=0, endtime=5)

        Select fiber segment from 250m to 1160m:
        >>> das.data_select(startlength=250, endlength=1160)

        Select specific time and space window:
        >>> das.data_select(starttime=10, endtime=15,
        ...                 startlength=500, endlength=800)

        Notes
        -----
        - Conversion from meters/seconds to indices uses self.dx and self.dt
        - Useful for reducing data volume before computationally expensive operations
        - Can be used iteratively to progressively narrow analysis window
        """
        if starttime is not None:
            i_starttime = int(starttime/self.dt)
        else:
            i_starttime = int(0)
        
        if endtime is None or endtime == -1:
            i_endtime = int(self.traces.shape[-1] - self.traces.shape[-1]%self.dt)
        else:
            i_endtime = int(endtime/self.dt)
        
        
        if startlength is None:
            i_startlength = int(0)
        else:
            i_startlength = int(startlength/self.dx)
        
        if endlength is None or endlength == -1:
            i_endlength = int(self.traces.shape[0])
        else:
            i_endlength = int(endlength/self.dx)
        
        traces = self.traces[i_startlength:i_endlength,i_starttime:i_endtime]
        self.traces = traces
        self.ntrs, self.npts = traces.shape
        
     
    def __trace_normalization(self, data):
        from scipy.signal  import detrend
        data = detrend(data, type='constant')
        nf = np.abs(data).max(axis=1)
        data = data / nf[:, np.newaxis]
        return data
    
        
    def visualization(self, path_results='./report/', savefig=False, fig_format='png'):
        '''
        This method creates a matplotlib figure of the DAS object.

        Parameters
        ----------
        path_results : String, optional
            Path where to store the results. The default is './report/'.
        savefig : Boolean, optional
            The default is False.
        fig_format : String, optional
            Format in which to save the figure. The default is png.

        Returns
        -------
        A matplotlib figure of the DAS object.

        '''
        time = np.arange(self.npts)*self.dt
        depth = np.arange(self.ntrs)*self.dx
        plt.figure(figsize=[10,5])
        plt.imshow(self.traces,
                   extent=[min(time), max(time), max(depth), min(depth)],
                   cmap='seismic',
                   aspect='auto')
        
        plt.ylabel('Distance along the fiber [m]')
        plt.xlabel('Relative time [s]')
        plt.tight_layout()
        
        if savefig is not False:
            plt.savefig(path_results + self.fname + '_imshow.' + fig_format, dpi=300)
        
   
    def __fk_filt(self, data, k0=False, low_vel_events=False):
        """
        Apply frequency-wavenumber (FK) domain filtering (private method).

        This method transforms data to FK domain and attenuates coherent noise:
        - k0 linear events (fiber-parallel noise, e.g., cable ringing)
        - Low-velocity hyperbolic events (e.g., ground roll, slow surface waves)

        ⚠️ WARNING: Parameters are tuned for FORGE 2019 dataset. Adjust for other datasets.

        Parameters
        ----------
        data : numpy.ndarray
            Input DAS data with shape (ntrs, npts).
        k0 : bool, optional
            If True, attenuates energy at k=0 wavenumber (linear events parallel
            to fiber). Applies soft taper (0.5 factor) to 3 outermost wavenumber
            bins and zeros 2 innermost bins. Default: False.
        low_vel_events : bool, optional
            If True, attenuates low-velocity hyperbolic events using triangular
            windowing in FK domain. Targets slow-moving coherent noise.
            Default: False.

        Returns
        -------
        numpy.ndarray
            FK-filtered data with shape (ntrs, npts).

        Notes
        -----
        **FORGE-specific hard-coded parameters:**
        - `max_value_outer_trian = int(m/2.5)`: Triangle window scaling factor
          * Adjust divisor (2.5) based on dominant wavenumber content
          * Larger divisor = narrower filter, removes less low-k energy
        - `delta_filt = int(10*signal_len)`: FK filter bandwidth
          * Multiplier (10) controls low-velocity rejection zone width
          * Larger value = more aggressive filtering of slow events
        - k0 filter: 3-trace boundary taper, 2-trace center rejection
          * May need adjustment for different channel spacing (dx)

        **Adaptation guidelines:**
        1. Visualize FK spectrum with plotfk() before filtering
        2. Identify dominant noise wavenumber/frequency ranges
        3. Adjust max_value_outer_trian divisor to match noise characteristics
        4. Tune delta_filt multiplier based on event velocity distribution
        5. Test on subset before applying to full dataset

        Algorithm:
        1. 2D FFT to FK domain (rfft2 for real input)
        2. Create multiplicative filter array (ones = pass, zeros = reject)
        3. Apply triangular window for low-velocity event rejection
        4. Apply k=0 rejection for fiber-parallel noise
        5. Multiply FK spectrum by filter (preserve phase)
        6. Inverse FFT back to space-time domain

        Examples
        --------
        Remove both k0 and low-velocity events:
        >>> filtered = self.__fk_filt(data, k0=True, low_vel_events=True)

        Remove only fiber-parallel noise:
        >>> filtered = self.__fk_filt(data, k0=True, low_vel_events=False)

        Warnings
        --------
        - Aggressive filtering may attenuate signal energy
        - Always compare filtered/unfiltered data visually
        - FK filtering works best with trace-normalized data
        """
        from scipy.signal import windows

        # Transform to FK domain
        fk = np.fft.rfft2(data)
        n, m = fk.shape
        filt = np.ones([n,m])  # Initialize pass-all filter

        ## FORGE-SPECIFIC: Define triangular window for low-velocity event rejection
        ## Adjust divisor (2.5) based on your data's wavenumber characteristics
        max_value_outer_trian = int(m/2.5)
        outer_window = (windows.triang(n) * max_value_outer_trian)

        signal_len = (self.npts/self.sampling_rate)
        ## FORGE-SPECIFIC: Filter bandwidth - adjust multiplier (10) for your dataset
        delta_filt = int(10*signal_len)

        # Apply k0 filtering (fiber-parallel linear events)
        if k0:
            filt[0:3,:] = 0.5    # Taper outermost 3 wavenumber bins
            filt[n-3:,:] = 0.5

        # Apply low-velocity event filtering (hyperbolic noise)
        if low_vel_events:
            for i in range(filt.shape[0]):
                # Create rejection zone using triangular window
                filt[i,int(outer_window[i])-int(delta_filt):int(outer_window[i])] = 0.5
                filt[i,:int(outer_window[i])-3] = 0.  # Zero low-k region

        # Strengthen k0 rejection (zero innermost bins)
        if k0:
            filt[0:2,:] = 0.
            filt[n-2:,:] = 0.

        # Apply filter preserving phase information
        fkfilt = np.abs(fk)*filt*np.exp(1j*np.angle(fk))
        data_filt = np.fft.irfft2(fkfilt)

        self.traces = data_filt
        return data_filt
        
            
    def plotfk(self, path_results='./report/', savefig=False, fig_format='png'):
        '''
        This method creates a Frequency-Wavenumber plot of the DAS object.

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
        A Frequency-Wavenumber plot of the DAS object.

        '''
        from matplotlib import colors
        f = np.fft.rfftfreq(self.npts, d=self.dt)
        k = np.fft.fftfreq(self.ntrs, d=self.dx)
        fk= np.fft.rfft2(self.traces) + 1
        fk = np.abs(fk) / np.max(np.abs(fk))
        
        plt.figure()
        plt.imshow(np.abs(np.fft.fftshift(fk, axes=(0,))).T,
                   extent=[min(k), max(k), min(f), max(f)],
                   aspect='auto',
                   cmap='plasma',
                   interpolation=None,
                   origin='lower',
                   norm=colors.LogNorm())
        
        h = plt.colorbar()
        h.set_label('Amplitude Spectra  (rel. 1 $(\epsilon/s)^2$)')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Wavenumber [1/m]')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        
        if savefig is True:
            plt.savefig(path_results + self.fname + '_FK.' + fig_format, dpi=300)
            
                
    def analytic_signal(self,traces):
        tracef=np.fft.fft(traces)
        nsta,nfreq=np.shape(tracef)
        freqs=np.fft.fftfreq(nfreq,self.dt)
        traceh=tracef+(np.sign(freqs).T*tracef)
        traces=traces+1j*np.fft.ifft(traceh).real
        self.envelope=np.abs(traces)

    # def stalta(self, traces):
    #     tsta = .01
    #     tlta = tsta*10
    #     nsta = int(tsta/self.dt)
    #     nlta = int(tlta/self.dt)
    #     ks = self.dt/tsta
    #     kl = self.dt/tlta
    #     sta0 = np.mean(traces[:,nlta:nlta+nsta]**2, axis=1)
    #     lta0 = np.mean(traces[:,0:nlta]**2, axis=1)
    #     stalta = np.zeros(np.shape(traces))
    #     for i in range(nlta+nsta,self.npts):
    #         sta0 = ks*traces[:,i]**2+((1.-ks)*sta0)
    #         lta0 = kl*traces[:,i-nsta]**2+((1.-kl)*lta0)
    #         stalta[:,i] = sta0/lta0
    #     stalta[:,0:nlta+nsta] = stalta[:,nlta+nsta:2*(nlta+nsta)]
    #     stalta = self.__trace_normalization(stalta)
    #     self.traces = stalta
        
    #     self.ntrs,self.npts = stalta.shape
    #     return stalta
    
    def stalta(self, traces):
        """
        Compute STA/LTA (Short-Term Average / Long-Term Average) characteristic function.

        This method calculates a classic seismic detection function that highlights
        transient signals by comparing short-term energy to long-term background energy.
        Uses recursive computation for efficiency.

        ⚠️ NOTE: Window lengths are hard-coded (tsta=0.01s, tlta=0.1s). Adjust for your data.

        Parameters
        ----------
        traces : numpy.ndarray
            Input DAS data with shape (ntrs, npts).

        Returns
        -------
        numpy.ndarray
            STA/LTA characteristic function, normalized to [-1, 1], shape (ntrs, npts).

        Notes
        -----
        **Hard-coded parameters (adjust for your dataset):**
        - `tsta = 0.01 s`: Short-term window length (10 ms)
          * Shorter = more sensitive to onset sharpness
          * Typical range: 0.01-0.05 s
        - `tlta = tsta * 10 = 0.1 s`: Long-term window length (100 ms)
          * Longer = more stable background estimate
          * Typical ratio: tlta = 10-20 × tsta

        **Algorithm:**
        1. Convert time windows to sample counts (nsta, nlta)
        2. Initialize STA/LTA with initial energy estimates
        3. Recursively update using exponential decay:
           - sta(i) = ks × E(i) + (1-ks) × sta(i-1)
           - lta(i) = kl × E(i-nsta) + (1-kl) × lta(i-1)
        4. Compute ratio: stalta(i) = sta(i) / lta(i)
        5. Fill initial samples with mirrored values
        6. Normalize to [-1, 1] range

        **Usage context:**
        - Can be used as preprocessing for semblance detection
        - Highlights transients but loses polarity information
        - Useful when radiation pattern effects are problematic

        Examples
        --------
        >>> cf = das.stalta(das.traces)
        >>> # cf now contains normalized STA/LTA characteristic function

        Warnings
        --------
        - Hard-coded windows may not suit all event types
        - For very short events (< 10 ms), reduce tsta
        - For high-noise data, increase tlta for stable background
        - Operation modifies self.traces IN-PLACE
        """
        # HARD-CODED PARAMETERS - adjust for your dataset
        tsta = .01          # Short-term window: 10 ms
        tlta = tsta*10      # Long-term window: 100 ms

        nsta = int(tsta/self.dt)
        nlta = int(tlta/self.dt)
        ks = self.dt/tsta   # STA decay factor
        kl = self.dt/tlta   # LTA decay factor

        # Initialize with energy in initial windows
        sta0 = np.mean(traces[:,nlta:nlta+nsta]**2, axis=1)
        lta0 = np.mean(traces[:,0:nlta]**2, axis=1)
        stalta = np.zeros(np.shape(traces))

        # Recursive STA/LTA computation
        for i in range(nlta+nsta,self.npts):
            sta0 = ks*traces[:,i]**2+((1.-ks)*sta0)
            lta0 = kl*traces[:,i-nsta]**2+((1.-kl)*lta0)
            stalta[:,i] = sta0/lta0

        # Fill initial samples with mirrored values
        stalta[:,0:nlta+nsta] = stalta[:,nlta+nsta:2*(nlta+nsta)]

        # Normalize to [-1, 1]
        stalta = self.__trace_normalization(stalta)
        self.traces = stalta

        self.ntrs,self.npts = stalta.shape
        return stalta
    
    