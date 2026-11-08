# ARCHIVIST
---------------------------------

Author: L. Bruno on behalf of NC team

Affiliation: INAF-IRA

Last update: 22/05/2026

Contact: l.bruno@ira.inaf.it


The "miniARCHIVIST.py" pipeline is a simplified version of the "ARCHIVIST.py" pipeline (Bruno et al. 2026) for public distribution and usage. ARCHIVIST is used to prepare the data products of the NC-FRB project, which are made public through INCART: 
https://ngc-frb-incart.ira.inaf.it/
miniARCHIVIST implements a small subset of steps from ARCHIVIST.py, allowing users to carry out the analysis of NC-FRBs. Specifically, miniARCHIVIST performs steps 4-6 as described in Sect. 3.1 of Bruno et al. 2026:
https://www.cambridge.org/core/journals/publications-of-the-astronomical-society-of-australia/article/northern-cross-fast-radio-burst-project-vi-the-incart-public-database/F0A45F096F2C7B5831BBDA94F3FE1D0F



#################################

REQUIRED PARAMETERS

1) f --> dataset in filterbank format.

2) DM --> Fiducial dispersion measure (in units of pc/cm3). 

3) W --> Fiducial FWHM of burst (in units of ms).  


Notes: DM and W can be obtained from standard pipelines for FRB search (e.g. Heimdall) or from INCART. These are initial representative guesses; W is then refined during fitting. Differently from ARCHIVIST, miniARCHIVIST does not perform the time-DM analysis. 


#################################


OPTIONAL PARAMETERS

1) rTOA --> Time of arrival of the FRB relative to the beginning of the observation (i.e. seconds after the starting time). For the NC-FRB cutout datasets, the burst occurs in the middle of the observation (ex: if the total observation length is 2 s, TOA is 1).    

2) fbeam --> Correction factor for beam attenuation, which is reported in the logfile in INCART. Default is 1, that is no correction is done. 

3) Ncyl --> Number of working cylinders during the observation, as reported in the logfile in INCART. Default is 16.  

4) twin --> Time window (in units of ms). If not set, it will be automatically computed to optimize the SNR. 

5) Ndown --> Number of temporal bins used for downsampling within the time window. If not set, it will be automatically computed.

6) freqshape --> Number of channels used for frequency downsampling for inspection purposes only. Default is 32.

7) mask_offset --> Centre of temporal mask. Set it in the case of multiple close bursts to reduce contamination. Set it as the offset (i.e. +/-N seconds) with respect to TOA of the burst that you aim to fit. 

8) mask_w --> Width of temporal mask (in units of s). Set it in combination with mask_offset.  

9) mask_offsetB --> Same as for mask_offset, but for a second mask to be added. 

10) mask_wB --> Same as for mask_w, but for a second mask to be added. 


#################################

OUTPUT

- plot showing the fitted timeseries and dynamic spectrum
- a logfile summarising the results

Key fitted parameters are: width, flux density of the peak, and "rectangular" fluence. 




  
