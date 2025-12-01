# peak_detection
Algorithm to identify extreme atmospheric moistening and drying based on retrieved IWV from ground-based microwave radiometer. 

Contact: christian.buhren (MSc. Christian Buhren @University of Cologne, Institute for Geophysics and Meteorology)

Required packages (version): Python (3.10.12), pandas (1.5.3), numpy (1.24.4), xarray (2023.6.0).

### Short description of the algorithm steps
First, local minima and maxima points are identified within a 12-hour rolling window. Amplitudes and durations between each minimum-maximum pair yield specific thresholds for extreme events. The threshold is calculated as the 95th percentile of typical amplitudes of the corresponding month within the study period, 2012-2024. Amplitudes exceeding this threshold are classified as extreme, with one detection for single distinct maxima (SP-M and SP-D). Instances of gradual moistening and drying over multiple steps are also identified and labeled as events with multiple detections (MP-M and MP-M).

