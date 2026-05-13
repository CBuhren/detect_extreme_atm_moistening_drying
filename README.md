# peak_detection

Algorithm to identify extreme atmospheric moistening and drying events based on integrated water vapor (IWV) retrieved from a ground-based microwave radiometer.

## Contact

Christian Buhren  
Institute for Geophysics and Meteorology, University of Cologne  
Email: christian.buhren

## Requirements

The algorithm was developed and tested with the following package versions:

- Python 3.10.12
- pandas 1.5.3
- numpy 1.24.4
- xarray 2023.6.0

## Input data

The script requires quality-controlled IWV data from the Humidity and Temperature Profiler (HATPRO) as input.  
The algorithm was developed for the related publication:

**DOI_PUBLICATION**

For the application at Ny-Ålesund, the following quality-controlled IWV dataset was used:

https://doi.pangaea.de/10.1594/PANGAEA.988284

## Algorithm overview

The algorithm identifies extreme atmospheric moistening and drying events from 10-minute resolved IWV time series.

First, local minima and maxima are detected within a 12-hour rolling window. IWV amplitudes and durations are then calculated for each minimum–maximum or maximum–minimum pair. Event-specific thresholds are derived from the 95th percentile of typical monthly IWV amplitudes during the study period from 2012 to 2024.

Amplitudes exceeding the monthly threshold are classified as extreme events. The algorithm distinguishes between:

- **CONT-M**: continuous moistening events with one distinct maximum
- **CONT-D**: continuous drying events with one distinct minimum
- **STEP-M**: stepwise moistening events with multiple detections
- **STEP-D**: stepwise drying events with multiple detections

A detailed description of the algorithm is provided in the related publication:

**DOI_PUBLICATION**

## Repository contents

### `define_event.py`

Main script to run the peak detection algorithm.

This script requires quality-controlled, 10-minute resolved IWV input data, for example from the Ny-Ålesund HATPRO dataset:

https://doi.pangaea.de/10.1594/PANGAEA.988284

File paths need to be adapted before running the script in a different environment.

### `Convert_df_events.py`

Secondary script used to generate the final event catalog.

This script checks all **STEP-M** and **STEP-D** detections, merges connected detections into single events, and derives updated IWV amplitudes and durations for these stepwise events.
