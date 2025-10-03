# HECTOR: coHerence-based Earthquake deteCTOR

A semblance-based microseismic event detector for Distributed Acoustic Sensing (DAS) data.

## Overview

HECTOR is a real-time microseismic event detection method that exploits the high spatial sampling density of DAS data. The detector identifies seismic events by computing waveform coherence along geometrical hyperbolic trajectories, independent of external velocity models or hypocentral locations.

**Key Features:**
- Model-independent detection using hyperbolic semblance analysis
- Optimized for DAS data with high spatial sampling (sub-meter to meter scale)
- Numba JIT compilation for computational efficiency
- Applicable to borehole and near-field monitoring scenarios
- Demonstrated to detect ~2× more events than standard STA/LTA methods

**Reference Publication:**
Porras, J., Pecci, D., Bocchini, G.M., et al. (2024). A semblance-based microseismic event detector for DAS data. *Geophysical Journal International*, 236(3), 1716-1727. https://doi.org/10.1093/gji/ggae016

---

## Methodology

### Detection Algorithm

HECTOR uses a two-stage approach:

#### 1. Waveform Coherence Analysis (Semblance Function)

The detector scans the DAS data using hyperbolic trajectories defined by:

```
t_i(T, X, C) = √(T² + (x_i - X)² / C²)
```

Where:
- **T**: Time offset (vertex position along time axis)
- **X**: Spatial offset (vertex position along fiber axis)
- **C**: Curvature coefficient (controls hyperbola shape)
- **x_i**: Position of the i-th DAS channel

The semblance function computes waveform coherence:

```
S(T, X, C) = [Σ(Σ A(t_ij))²] / [M × Σ(Σ A(t_ij)²)]
```

Where:
- **M**: Number of DAS traces (channels)
- **N**: Sample window length
- **A(t_ij)**: Amplitude at trace i and time t_ij

Semblance values range from 0 (no coherence) to 1 (perfect coherence).

#### 2. Clustering and Event Extraction

The coherence time-series undergoes:
1. **Threshold detection**: Events exceed a noise-based threshold
2. **Clustering**: Nearby detections grouped by temporal proximity
3. **SNR filtering**: Signal-to-noise ratio criterion (default: 6 dB threshold)
4. **Event timing**: Detection time corresponds to cluster start

### Data Preprocessing Workflow

The denoising workflow consists of:

1. **Low-pass filtering** (anti-aliasing before downsampling)
2. **Downsampling** (computational efficiency)
3. **Detrending** (remove linear and mean trends)
4. **Trace normalization** (correct for geometric spreading, coupling effects)
5. **Bandpass filtering** (isolate frequency band of interest)
6. **FK filtering** (attenuate coherent noise, especially k=0 linear events)

---

## Directory Structure

```
HECTOR/
├── das/                    # Core Python modules
│   ├── DAS.py             # Base class for DAS data handling
│   └── Detector.py        # Semblance-based detector (inherits from DAS)
├── input/                 # Input DAS data files
│   └── [Place your data files here]
├── report/                # Output directory
│   ├── [figures]          # Detection plots (.png, .pdf)
│   └── [events]           # Event catalogs (.out files)
├── docs/                  # Documentation and reference paper
├── HECTOR_example.ipynb   # Complete workflow demonstration
└── README.md
```

---

## Input Data Organization

### Supported File Formats

The `DAS` class supports four file formats:

| Format | Extension | Metadata Extraction | Duration Parameter |
|--------|-----------|---------------------|-------------------|
| **SEG-Y** | `.sgy`, `.segy` | Automatic | Not required |
| **TDMS** | `.tdms` | Automatic | Not required |
| **NumPy** | `.npy` | Manual | **Required** |
| **HDF5** | `.h5` | Manual | **Required** |

### Input File Placement

Place your DAS data files in the `input/` directory:

```bash
input/
├── your_data_file.sgy
├── another_file.tdms
└── synthetic_data.npy
```

### Required Metadata

When creating a Detector object, provide:

```python
from Detector import Detector

arr = Detector(
    file='./input/your_file.sgy',    # Full path to data file
    dx=1.02,                          # Channel spacing (meters)
    gl=10.0,                          # Gauge length (meters)
    fname='your_file',                # File name (for outputs)
    file_format='segy',               # Format: 'segy', 'tdms', 'npy', 'h5'
    duration=None                     # Duration in seconds (required for npy/h5)
)
```

**Data Array Format:**
- Shape: `(ntrs, npts)` where `ntrs` = number of traces (channels), `npts` = time samples
- Units: Strain rate (typical DAS output)

---

## Output Structure

### 1. Event Catalog Files (`.out`)

Located in `report/` directory, tab-separated format:

```
filename    relative_time(s)    absolute_timestamp
your_file   0.125              2022-04-21T13:48:22.125000Z
your_file   1.834              2022-04-21T13:48:23.834000Z
```

**Columns:**
- **filename**: Input data file name
- **relative_time**: Event time relative to file start (seconds)
- **absolute_timestamp**: Absolute UTC timestamp (for segy/tdms)

### 2. Detection Report Figures

Generated by `arr.plot_report(savefig=True)`:

**Multi-panel visualization:**
- **Panel 1**: Coherence time-series with detection threshold and event markers
- **Panel 2**: Semblance matrix (Curvature vs. Time)
- **Panel 3**: Raw/filtered DAS data with event overlay

**Output file:** `report/[filename]_detection_plot.png` (or `.pdf`)

### 3. Additional Visualization Outputs

- `[filename]_imshow.png`: DAS data visualization
- `[filename]_FK.png`: Frequency-wavenumber spectrum
- `[filename]_hyperbola_tuning.png`: Hyperbolic trajectory overlay (parameter tuning)

---

## Typical Workflow

### Example: Complete Detection Pipeline

```python
import sys
sys.path.append('./das/')
from Detector import Detector

# Step 1: Create Detector object
arr = Detector(
    file='./input/FORGE_data.sgy',
    dx=1.02,
    gl=10.0,
    fname='FORGE_data',
    file_format='segy',
    duration=None
)

# Step 2: Data selection (optional - spatiotemporal subset)
arr.data_select(
    starttime=0,      # Start time (seconds)
    endtime=5,        # End time (seconds)
    startlength=250,  # Start position along fiber (meters)
    endlength=1160    # End position along fiber (meters)
)

# Step 3: Preprocessing
arr.denoise(
    arr.traces,
    sampling_rate_new=500,    # Downsample to 500 Hz
    ftype='bandpass',         # Filter type
    fmin=1,                   # Min frequency (Hz)
    fmax=249,                 # Max frequency (Hz)
    k0=True,                  # Remove k=0 coherent noise
    low_vel_events=True       # Remove low-velocity events
)

# Step 4: Run detector
arr.detector(
    ns_window=20,       # Semblance window width (samples)
    a=20,               # Hyperbola vertex width parameter
    b_step=10,          # Time step (samples)
    c_min=70,           # Min curvature coefficient
    c_max=500,          # Max curvature coefficient
    c_step=5,           # Curvature step
    d_static=892,       # Static lateral position (traces) - OR use lat_search
    lat_search=False    # Set True for automatic lateral search
)

# Step 5: Extract events
events = arr.detected_events(
    min_numb_detections=10,   # Min samples above threshold to form cluster
    max_dist_detections=2     # Max consecutive samples below threshold in cluster
)

# Step 6: Generate report
arr.plot_report(savefig=True, fig_format='png')
```

---

## Key Parameters Explained

### Semblance Parameters

| Parameter | Description | Typical Range | FORGE Value |
|-----------|-------------|---------------|-------------|
| `ns_window` | Width of data window for semblance computation (samples) | 10-30 | 20 |
| `a` | Hyperbola vertex width (smaller = wider hyperbola) | 10-50 | 20 |
| `b_step` | Time step for scanning (samples) | 5-20 | 10 |
| `c_min` | Minimum curvature coefficient (smaller = higher curvature) | 50-100 | 70 |
| `c_max` | Maximum curvature coefficient | 200-500 | 500 |
| `c_step` | Curvature scanning step | 5-10 | 5 |
| `d_static` | Fixed lateral position (trace index) | Variable | 892 |
| `lat_search` | Enable automatic lateral position search | True/False | True |

**Notes:**
- Curvature parameters depend on source-fiber distance
- For near-field events: use smaller `c_min`, `c_max`
- For far-field events: use larger `c_max` or `lat_search=True`

### Clustering Parameters

| Parameter | Description | FORGE Value |
|-----------|-------------|-------------|
| `min_numb_detections` | Minimum samples above threshold to declare event | 10 |
| `max_dist_detections` | Max consecutive samples below threshold within cluster | 2 |
| SNR threshold | Minimum signal-to-noise ratio (dB) for event confirmation | 6 dB |

---

## FORGE-Specific Hard-Coded Parameters

### ⚠️ **Functions with FORGE Dataset Hard-Coding**

#### 1. **`DAS.__fk_filt()` (das/DAS.py:320-351)**

**Hard-coded parameters:**

```python
max_value_outer_trian = int(m/2.5)  # Line 328 - Triangle window scaling
delta_filt = int(10*signal_len)     # Line 332 - FK filter bandwidth
```

**FORGE-specific behavior:**
- Lines 335-336: k0 filtering (3 traces at boundaries set to 0.5)
- Lines 338-345: Low-velocity event filtering with triangular window

**Recommendation for new datasets:**
- Adjust `2.5` scaling factor based on dominant wavenumber range
- Tune `delta_filt` multiplier (currently `10`) based on event characteristics
- Visualize FK spectrum with `arr.plotfk()` before applying FK filter

#### 2. **`Detector.detected_events()` (das/Detector.py:450-585)**

**Hard-coded parameters:**

```python
# Line 468: Noise threshold calculation
noise = stats.trim_mean(coh_vector, 0.05)  # Trim 5% of extremes

# Line 528-530: SNR noise window for early detections
if detect_ <= 22:
    noise_det = noise * 2.0  # Special case for first 22 samples

# Line 532: Signal window length
signal_det = np.sqrt(np.mean(np.square(semb_coeff[:30])))  # 30 samples

# Line 536: SNR threshold
if SNR_det >= 6:  # 6 dB threshold

# Line 539: Long cluster threshold
if len(i) > 60:  # Check for multiple events in cluster

# Line 551: Second event SNR threshold
if SNR > 4:  # 4 dB for second event in cluster
```

**FORGE-specific values:**
- `0.05`: Trimmed mean percentage for noise estimation
- `22`: Boundary sample threshold for noise window adjustment
- `2.0`: Noise multiplier for early detections
- `30`: Signal window length (samples)
- `6 dB`: Primary SNR threshold
- `60`: Maximum cluster length before multi-event check
- `4 dB`: SNR threshold for second event

**Recommendation for new datasets:**
- Adjust SNR thresholds based on noise characteristics
- Modify cluster length threshold (`60`) based on expected event duration
- Change signal/noise window lengths based on sampling rate

#### 3. **`Detector.stalta()` (das/DAS.py:428-447)**

**Hard-coded parameters:**

```python
tsta = 0.01          # Line 429: Short-term average window (seconds)
tlta = tsta * 10     # Line 430: Long-term average window (seconds)
```

**Recommendation:**
- These are reasonable defaults but may need adjustment for different event types
- Typical range: `tsta = 0.01-0.05 s`, `tlta = 10×tsta` to `20×tsta`

---

## Parameter Tuning Guidelines

### For Different Monitoring Scenarios

#### **Near-field monitoring (events close to fiber)**
- Smaller curvature range: `c_min=50, c_max=200`
- Finer time step: `b_step=5`
- Higher SNR threshold if noise is low: `SNR≥8 dB`

#### **Far-field monitoring (events distant from fiber)**
- Larger curvature range: `c_min=100, c_max=800`
- Enable lateral search: `lat_search=True`
- Coarser time step for speed: `b_step=15-20`

#### **High-noise environments**
- Increase `min_numb_detections=15-20`
- Raise SNR threshold: `SNR≥8-10 dB`
- Apply more aggressive FK filtering

### Hyperbola Parameter Tuning

Use the `hyperbolae_tuning()` method to visualize scanning patterns:

```python
arr.hyperbolae_tuning(
    a=20,
    b=375,
    d=892,
    c_min=70,
    c_max=500,
    c_step=5,
    savefig=True
)
```

This overlays hyperbolic trajectories on your data to verify parameter coverage.

---

## Performance Considerations

### Computational Efficiency

**Optimization strategies implemented:**
- **Numba JIT compilation**: `sample_select()` function (Detector.py:18-31)
- **Downsampling**: Reduces data volume (e.g., 4000 Hz → 500 Hz)
- **Selective channel range**: Process only relevant fiber segments

**Benchmark (FORGE dataset):**
- Processing time: ~0.25 s per second of data
- Hardware: Intel i7 quad-core, 16 GB RAM
- Data: 1034 channels @ 500 Hz

**Data volume estimates:**
- FORGE 2022 campaign: 1.3 TB/day (4 kHz sampling, 2482 m fiber)
- With downsampling to 500 Hz: ~163 GB/day

### Lateral Search Trade-offs

Using `lat_search=True`:
- **Pros**: Automatic optimization, better for unknown source locations
- **Cons**: Computes N semblance matrices (N = number of X positions)
- **Recommendation**: Use `d_static` if source region is known

---

## Dependencies

Required Python packages:

```bash
pip install obspy numpy scipy numba matplotlib nptdms h5py wget
```

**Package purposes:**
- `obspy`: SEG-Y file reading, timestamp handling
- `numpy`: Array operations
- `scipy`: Signal processing (filtering, resampling)
- `numba`: JIT compilation for speed
- `matplotlib`: Visualization
- `nptdms`: TDMS file format support
- `h5py`: HDF5 file format support
- `wget`: Example notebook data download

---

## Limitations and Considerations

### Current Limitations

1. **Single-component data**: DAS measures only along fiber axis (no azimuthal information)
2. **Multi-event detection**: Maximum 2 events per cluster
3. **Straight fiber assumption**: Optimized for linear/borehole geometries
4. **No automatic location**: Provides detection times only (not hypocenters)

### Best Practices

- **Always visualize FK spectrum** before applying FK filter
- **Calibrate on known events** before processing full dataset
- **Inspect false detections** to refine clustering parameters
- **Use `lat_search=True`** for initial runs, then fix `d_static` if feasible
- **Save intermediate results** (preprocessed data) for parameter iteration

---

## Citation

If you use HECTOR in your research, please cite:

```bibtex
@article{porras2024hector,
  title={A semblance-based microseismic event detector for DAS data},
  author={Porras, Juan and Pecci, Davide and Bocchini, Gian Maria and
          Gaviano, Sonja and De Solda, Michele and Tuinstra, Katinka and
          Lanza, Federica and Tognarelli, Andrea and Stucchi, Eusebio and
          Grigoli, Francesco},
  journal={Geophysical Journal International},
  volume={236},
  number={3},
  pages={1716--1727},
  year={2024},
  publisher={Oxford University Press},
  doi={10.1093/gji/ggae016}
}
```

---

## License

See LICENSE file for details.

---

## Contact and Support

For questions, issues, or contributions:
- GitHub: https://github.com/juanucr/HECTOR
- Reference paper: https://doi.org/10.1093/gji/ggae016

---

## Acknowledgments

This work used data from the FORGE (Frontier Observatory for Research in Geothermal Energy) project, Utah, USA. HECTOR was developed by the University of Pisa (Department of Earth Sciences) in collaboration with the Swiss Seismological Service.
