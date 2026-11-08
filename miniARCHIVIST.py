import subprocess
import sys
import numpy as np
import os
from sigpyproc.base import Filterbank
from sigpyproc.readers import FilReader
from sigpyproc.header import Header
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse
import your
from skimage.transform import resize_local_mean as resize
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from astropy.time import Time


#run example: python3 miniARCHIVIST.py --f FRB20220912A_beam_1688095815.364019_processed_cutout_1217.0432s_to_1219.0432s.fil --DM 220 --W 3



###############################################
#argparse: mandatory
###############################################

parser = argparse.ArgumentParser(description = 'Output files for INCART')

parser.add_argument('--f', type=str, help = 'Filterbank file', required=True)
parser.add_argument('--DM', type=float, help = 'Fiducial DM (pc/cm3)', required=True)
parser.add_argument('--W', type=float, help = 'Fiducial Width (ms)', required=True)


###############################################
#argparse: optional
###############################################

parser.add_argument('--rTOA', type=float, help='TOA (s after start)', required=False)
parser.add_argument('--fbeam', type=float, help='Correction factor for beam attenuation (default 1)', required=False)
parser.add_argument('--Ncyl', type=int, help = 'Number of cylinders (default 16)', required=False)
parser.add_argument('--twin', type=float, help='Time window (ms)', required=False)
parser.add_argument('--Ndown', type=int, help='Number of downsampled samples', required=False)
parser.add_argument('--freqshape', type=int, help='Downsampled channels (default 32)', required=False)
parser.add_argument('--mask_offset', type=float, help='Offset (+-N sec wrt main burst) for manual mask', required=False)
parser.add_argument('--mask_w', type=float, help='Manual mask width (s)', required=False)
parser.add_argument('--mask_offsetB', type=float, help='Offset (+-N sec wrt main burst) for additional manual mask', required=False)
parser.add_argument('--mask_wB', type=float, help='Additional manual mask width (s)', required=False)



args = parser.parse_args()


filfile = args.f
candDM = args.DM
inp_W = args.W

if args.fbeam:
    fbeam = args.fbeam
else:
    fbeam = 1.

logfile = "logfile.txt"


###############################################
#save header of uncut file in txt file
###############################################


def extract_original_header(filfile):

    x = subprocess.check_output(['header', filfile], text=True)
    print(x)
    with open(logfile, "w") as f:
        f.write(x)
    
    return



###############################################
#extract required header info for processing
###############################################


def out_info(keyword):

    
    head = {}

    with open(logfile, 'r') as f:
        for line in f:
            key, value = line.split(':', 1)
            head[key.strip()] = value.strip()

    return head[keyword]



###############################################
#update header with new info
###############################################

def in_info(keyword, value):
    
    with open(logfile, 'a') as f:
        f.write(f"{keyword}: {value}\n")
    
    return



###############################################
#delete existing keyword
###############################################

def delete_info(keyword):

    head = {}

    with open(logfile, 'r') as f:
        for line in f:
            key, value = line.split(':', 1)
            head[key.strip()] = value.strip()
    
    del head[keyword]
    
    with open(logfile, 'w') as f:
        for key, value in head.items():
            f.write(f"{key}: {value}\n")
    
    return



    
###############################################
#Manual masking (in case of multiple close bursts)
###############################################

def manual_mask(mask_offset, mask_w, t0=0):

    mask_start = (t0 + float(mask_offset)) - mask_w / 2
    mask_end = (t0 + float(mask_offset)) + mask_w / 2

    return round(mask_start,4), round(mask_end,4)


###############################################
#Fit functions
###############################################

#compute optimal timewindow based on expected Width
def optimal_timewindow(inp_W):
 
    if args.twin:
        timewin = args.twin

    else:
        if inp_W < 3.0:
            timewin = 100
        elif inp_W <= 6.0 and args.mask_offset:
            timewin = 400
        elif 3.0 <= inp_W <= 6.0:
            timewin = 200
        elif 6.0 < inp_W < 40.0:
            timewin = 400
        else:
            timewin = 800

 
    return timewin



#compute optimal temporal downsampling based on timewindow and Width
def optimal_timeshape(timewin, inp_W):

    if args.Ndown:
        Ndown = args.Ndown
    
    elif inp_W >= 30.:
        Ndown = 1024
        
    elif inp_W < 3.:
        Ndown = 128
    
    else:
        k = 1.5 #from Nyquist theorem (1.2 higher res with lower SNR; 2 lower res with higher SNR)
        ratio = timewin / (k * inp_W )
        
        Ndown = 2 ** int(np.floor(np.log2(ratio)))
        Ndown = max(Ndown, 256)
    
    
    
    in_info("Downsampled bins", Ndown)
    in_info("Temporal window (ms)", timewin)
    
#    if inp_W < 2:
#        Ndown = 512
    
    return Ndown
  
  

    

#fit flux and fluence
def compute_flux_fluence(SNpeak, W, BW, xi):
    
    if args.Ncyl:
        num_cyl = args.Ncyl
    else:
        num_cyl = 16
    in_info("Number of cylinders", num_cyl)
    
    A = num_cyl * 4
    Np = 1 #number of polarisation
    SEFD = 8400. #Jy ( +- 420 --> 5%; ref: Trudu+22 )
    downsampled_time = float(out_info("Downsampled time (us)")) * 1.e-6
    
    therm_noise = SEFD / (A * np.sqrt( Np * BW*1.e6 * downsampled_time * (1-xi) ) ) #Jy
    
    
    flux_peak = SNpeak * therm_noise * fbeam #Jy
    err_rel_amp = float(out_info("Err. Norm. Gauss Amp.")) / float(out_info("Norm. Gauss Amp."))
    err_rel_W = float(out_info("Err. Fit Width (ms)")) / float(out_info("Fit Width (ms)"))
    err_flux_peak = flux_peak * err_rel_amp


    fluence_from_peak = flux_peak * W*1.e3 #Jy ms
    err_fluence_from_peak = fluence_from_peak * np.sqrt( (err_flux_peak/flux_peak)**2 + err_rel_W**2 )
    
    print("Flux density peak: "+str(flux_peak)+" Jy")
    print("Rectangular fluence: "+str(fluence_from_peak)+" Jy ms")
    
    
    in_info("Flux density peak (Jy)", flux_peak)
    in_info("Err. Flux density peak (Jy)", err_flux_peak)
    in_info("Rectangular Fluence (Jy ms)", fluence_from_peak)
    in_info("Err. Rectangular Fluence (Jy ms)", err_fluence_from_peak)
    
    
    
    return
    

#Gaussian profile
def gauss(x, x0, sigma, Amp):

    return Amp * np.exp( -(x-x0)**2 / (2 * sigma**2) )



#Fit burst with Gaussian profile to get width = FWHM
def get_width(time, timeseries, tmax, ax):
    
    if args.mask_offset and args.mask_w:
        mask_start, mask_end = manual_mask(args.mask_offset, args.mask_w)
        mask = (time < mask_start) | (time > mask_end)
        time = time[mask]
        timeseries = timeseries[mask]
        if args.mask_offsetB and args.mask_wB:
            mask_startB, mask_endB = manual_mask(args.mask_offsetB, args.mask_wB)
            maskB = (time < mask_startB) | (time > mask_endB)
            time = time[maskB]
            timeseries = timeseries[maskB]

        
    par_time_opt, par_time_cov = curve_fit(gauss, time, timeseries, p0 = [tmax, inp_W / (2.355e3), 1])
    par_err = np.sqrt(np.diag(par_time_cov))
    
    
    sigma_t = np.abs(par_time_opt[1])

    ax.plot(time, gauss(time, *par_time_opt), color='red', linewidth=4)
    
    W = 2.355 * sigma_t
    errW = par_err[1] * 2.355

    print("FWHM width: "+str(W * 1.e3)+" ms")


    
    in_info("Norm. Gauss Amp.", par_time_opt[2])
    in_info("Err. Norm. Gauss Amp.", par_err[2])
    in_info("Fit Width (ms)", W * 1.e3)
    in_info("Err. Fit Width (ms)", errW * 1.e3)
    in_info("Fit gauss center (ms wrt TOA)", par_time_opt[0] * 1.e3)
    
    
    
    return W



def get_snr(time, timeseries):


    W = float( out_info("Fit Width (ms)") ) * 1.e-3 #s
    sigma = W / 2.355 #ms
    x0 = float( out_info("Fit gauss center (ms wrt TOA)") ) * 1.e-3 #s
    print('Gauss centre (ms)', x0*1000)
    Amp = float( out_info("Norm. Gauss Amp.") ) #pseudoJy/ms
    
    tsamp = float( out_info("Sample time (us)") ) * 1.e-6
    
    
    half_width_bins = int((2 * sigma) / tsamp) #consider out of 2sigma as noise

    if args.mask_offset and args.mask_w:
        mask_start, mask_end = manual_mask(args.mask_offset, args.mask_w)
        in_info("Mask in", "["+str(mask_start)+","+str(mask_end)+"]")
        mask = (time < mask_start) | (time > mask_end)
        time = time[mask]
        timeseries = timeseries[mask]
        if args.mask_offsetB and args.mask_wB:
            mask_startB, mask_endB = manual_mask(args.mask_offsetB, args.mask_wB)
            in_info("Mask2 in", "["+str(mask_startB)+","+str(mask_endB)+"]")
            maskB = (time < mask_startB) | (time > mask_endB)
            time = time[maskB]
            timeseries = timeseries[maskB]


    peak_idx = np.argmax(timeseries)
    left = max(0, peak_idx - half_width_bins)
    right = min(len(timeseries), peak_idx + half_width_bins)

    snr_mask = np.ones_like(timeseries, dtype=bool)
    snr_mask[left:right] = False #mask burst
    
    std = np.std(timeseries[snr_mask]) #std of noise
    
    
    k_integration = 0.9545 #[x0-2sigma; x0+2sigma]
    area = Amp * np.sqrt(2 * np.pi) * sigma * k_integration #Gauss area
    
     
    timewindow = optimal_timewindow(W * 1e3) #ms
    timeshape =  optimal_timeshape(timewindow, W * 1e3) #temporal bin number


    if args.mask_offset and args.mask_w:
        timeshape = mask.sum()
    fdown = np.sqrt( W * 1.e3 * timeshape / timewindow ) #downsampling rescaling factor
    in_info("Downsampling factor", fdown)
    
    downsampled_time = 1.e-3 * timewindow / timeshape
    in_info("Downsampled time (us)", downsampled_time * 1.e6)

    SNRpeak = Amp  / ( std )
    SNRint = area / (std * fdown * downsampled_time) #integrated SNR
    
    print(f"Integrated SNR: {SNRint:.2f}")

    in_info("SNR peak", SNRpeak)
    in_info("SNR integrated", SNRint)
  
    chisq = np.sum( ( (timeseries - gauss(time, x0, sigma, Amp) ) / std )**2 )
    dof = len(time) - 3 #Ndata- Npar
    print("Gauss fit Chi2red: ", round(chisq/dof, 2))
    in_info("Gauss fit Chi2red", chisq/dof)

    return SNRpeak
    


#Compute dispersion delay (units: seconds) t_DM between f_start and f_stop
def dispersion_delay(fstart, fstop, dms = None):

    return ( 4148808.0e-3 * dms * (1 / fstart ** 2 - 1 / fstop ** 2))


#Renormalize the data to a smoothed baseline
def renormalize_data(array, badchans_mask = None):

    renorm_data = np.copy(array)
    spec = renorm_data.mean(1)
    if badchans_mask is None:
        badchans_mask = np.zeros(spec.size, dtype=bool)

    for i, newd in enumerate(renorm_data):
        renorm_data[i, :] = (newd - spec[i]) / spec[i]

    baseline = np.mean(renorm_data[~badchans_mask, :], axis=0)
    badbins = np.zeros(baseline.size, dtype=bool)

    if baseline.size > 1000:
        smooth_baseline = gaussian_filter(baseline, 101, truncate = 1)
        detr = baseline - smooth_baseline

        ordered = np.sort(detr)
        q1 = ordered[baseline.size // 4]
        q2 = ordered[baseline.size // 2]
        q3 = ordered[baseline.size // 4 * 3]
        lowlim = q2 - 2 * (q2 - q1)
        hilim = q2 + 2 * (q3 - q2)

        badbins = (detr < lowlim) | (detr > hilim)
        baseline = smooth_baseline

    renorm_data[~badchans_mask, :] -= baseline
    renorm_data[badchans_mask, :] = 0
    renorm_data[:, badbins] = 0

    return renorm_data




#Compute dedispersion of a waterfall matrix given an input DM
def dedisperse(wfall, DM, freq, dt):

    k_DM = 4148.808
    dedisp = np.zeros_like(wfall)

    # pick reference frequency for dedispersion
    reference_frequency = freq[0]

    shift = (k_DM * DM * (reference_frequency**-2 - freq**-2) / dt).round().astype(int)
    for i, ts in enumerate(wfall):
        dedisp[i] = np.roll(ts, shift[i])

    return dedisp



#Set spectrogram for plotting: re-normalize, flag bad channels, dedispersion
def prepare_data(data, tstart, tstop, badchans, freqs, tsamp, DM):

    data = np.asarray(data , dtype = float)
    
    
    data = renormalize_data(data, badchans_mask = badchans)
    data[badchans, :] = 0
    
    dedispdata = dedisperse(data, DM, freqs, tsamp)
    dedispdata[badchans, :] = 0

    return badchans, dedispdata


# Downsample an RFI mask. It considers group of channels and takes as a new downsampled value the most common value between 0 or 1.
def downsample_mask(badchans, newshape):

    oldshape = badchans.shape[0]

    ratio = int(oldshape / newshape)

    downbadchans = np.zeros((newshape,), dtype = int)

    badchansn = np.zeros((oldshape,) , dtype = int)
    badchansn[badchans] = 1

    for k in range(newshape):

        values, counts = np.unique(badchansn[k * ratio : (k+1) * ratio], return_counts = True)
        ind = np.argmax(counts)
        downbadchans[k] = values[ind]

    downbadchans = np.asarray(downbadchans, dtype = bool)
    
    return downbadchans



def make_plot(filname, candDM):

    if args.freqshape:
        freqshape = args.freqshape
    if not args.freqshape:
        freqshape = 32
 
 
    filfile = your.Your(filname)   # Fit will be done on cut file based on original header

    tsamp     = filfile.your_header.native_tsamp    # sampling time
    nsamp     = filfile.your_header.native_nspectra    # number of time samples
    nchan     = filfile.your_header.native_nchans   # number of channels
    foff      = filfile.your_header.foff    # channel bandwidth
    fch0      = filfile.your_header.fch1    # frequency of first channel (top band)
    
    bw = nchan * foff
    fc = fch0 + bw/2
    obslen = nsamp * tsamp
    
    freqs = np.arange(fc - bw / 2, fc + bw / 2, foff)
    times = np.linspace(0, obslen, int(nsamp))
    channels = np.arange(nchan)
    Delta_t = np.abs(dispersion_delay(freqs[-1], freqs[0], dms = candDM))

    if args.rTOA:
        candTC = args.rTOA
    elif not args.rTOA:
        candTC = round(obslen / 2, 4)
    
    
    boxcar = 256
    tstart  = candTC - 1.0 * Delta_t + boxcar * tsamp
    tstop   = candTC + 1.0 * Delta_t + boxcar * tsamp
    nstart = np.rint(tstart / tsamp).astype(int)
    nstop  = np.rint(tstop  / tsamp).astype(int)
    ncand  = np.rint(candTC  / tsamp).astype(int)
    bw = np.abs(bw)

    if not args.rTOA:
        data = filfile.get_data(nstart = 0, nsamp = int(nsamp))
    elif args.rTOA:
        data = filfile.get_data(nstart = 0, nsamp = 8192)


    badchans = your.utils.rfi.sk_filter(data, foff, nchan, tsamp, sigma = 5)

    data = filfile.get_data(nstart = nstart, nsamp = nstop - nstart)
    data = data.T
    badchans, dedispdata = prepare_data(data, tstart, tstop, badchans, freqs, tsamp, candDM)


    nbchans = np.count_nonzero(badchans)
    xi = nbchans / nchan
    in_info("SK masking fraction", xi)


    max = ncand - nstart

    timewin = optimal_timewindow(inp_W)
    twin = timewin * 1e-3 / 2
    win  = np.rint(twin / tsamp).astype(int)


    print("Performing dedispersion at "+str(candDM)+" pc cm^-3")
    
    dedispdata = dedispdata[:, max - win : max + win]
    dedispdata = np.array(dedispdata, dtype = float)

  
    timeshape = optimal_timeshape(timewin, inp_W)
    
    dedispdata = resize(dedispdata, (freqshape, timeshape ))

    if freqshape < nchan:

        badchans = downsample_mask(badchans, freqshape)
    

    lc = dedispdata.mean(0)
    lc = lc / np.max(lc)
    timeplot = np.linspace(-twin , +twin , dedispdata.shape[1])



    fig = plt.figure(figsize=(12, 12))
    mpl.rcParams['axes.linewidth'] = 1.0
    plt.subplots_adjust(top=0.99, bottom=0.08, right=0.95, left=0.1)

    #Plot time series, dynamic spectrum, butterfly
    gs = plt.GridSpec(3, 1, height_ratios=[4, 6, 6], hspace=0.05)
    ax0 = plt.subplot(gs[0, 0])
    ax1 = plt.subplot(gs[1, 0])
    
    
    ax0.margins(x=0)
    ax0.tick_params(labelsize=25, axis='both', length=8, width=4)
    ax0.tick_params(axis='x', labelbottom=False)
    ax0.set_ylabel("Norm. Signal", fontsize=25)

    #ax1.tick_params(labelsize=25)
    ax1.tick_params(labelsize=25, axis='both', length=8, width=4)
    ax1.tick_params(axis='x', labelbottom=True)
    ax1.set_ylabel("$\\nu$ (MHz)", fontsize=25)
    ax1.set_xlabel("t (ms)" , fontsize=25)
  
    fig.align_labels()

    new_dt = np.abs(timeplot[0] - timeplot[1])

    if new_dt < tsamp:
        new_dt = tsamp
        
    
    W = get_width(timeplot, lc, timeplot[np.rint(len(timeplot) / 2).astype(int)], ax0)
    wsamp = np.rint(W / new_dt).astype(int)


    SNR = get_snr(timeplot, lc)
    compute_flux_fluence(SNR, W, bw, xi)

    ax0.step(timeplot, lc, where='mid', linewidth = 2, color = "black")

    U = 1000
    maskmap =  mpl.colors.ListedColormap(["white", "red"])
    whitemap = mpl.colors.ListedColormap(["white", "none"])
    ones = np.ones(dedispdata.shape, dtype=bool)
    mask = np.array([badchans,]*timeshape).T
    e = np.rint(dedispdata.shape[1] / 25).astype(int)
    ones[:, e : - e] = 0
    channels   = np.ma.MaskedArray(ones, mask = ~mask)
    
        


    # AX1: dynamic spectrum
    ax1.imshow(dedispdata, aspect="auto", extent=(timeplot[0]*U, timeplot[-1]*U, freqs[-1], freqs[0]), cmap="gist_heat", vmin=np.percentile(dedispdata, 5), vmax=np.max(dedispdata))
    ax1.imshow(channels, aspect="auto", extent=(timeplot[0]*U, timeplot[-1]*U, freqs[-1], freqs[0]), cmap=maskmap)

    if args.mask_offset and args.mask_w and not args.mask_offsetB and not args.mask_wB:
        mask_start, mask_end = manual_mask(args.mask_offset, args.mask_w)
        t_mask = (timeplot < mask_start) | (timeplot > mask_end)
        tempmask_2d = np.tile(t_mask, (dedispdata.shape[0], 1))
        ax1.imshow(tempmask_2d, aspect="auto", extent=(timeplot[0]*U, timeplot[-1]*U, freqs[-1], freqs[0]), cmap=whitemap, alpha=1.0, interpolation='none')
    elif args.mask_offset and args.mask_w and args.mask_offsetB and args.mask_wB:
        mask_start, mask_end = manual_mask(args.mask_offset, args.mask_w)
        mask_startB, mask_endB = manual_mask(args.mask_offsetB, args.mask_wB)
        masked_region = (((timeplot >= mask_start) & (timeplot <= mask_end)) | ((timeplot >= mask_startB) & (timeplot <= mask_endB)))
        t_mask = ~masked_region
        tempmask_2d = np.tile(t_mask, (dedispdata.shape[0], 1))
        ax1.imshow(tempmask_2d, aspect="auto", extent=(timeplot[0]*U, timeplot[-1]*U, freqs[-1], freqs[0]), cmap=whitemap, alpha=1.0, interpolation='none')


       
    plt.savefig(str(filname)+".png", bbox_inches='tight')

    plt.close('all')






"""
MAIN CODE
"""

#step1: header

extract_original_header(filfile)


#step 2: compute beam correction

make_plot(str(filfile), candDM)


#step 3: update logfile


key_to_delete = ["Header size (bytes)", "Data size (bytes)", "Start AZ (deg)", "Start ZA (deg)", "Gregorian date (YYYY/MM/DD)", "Number of samples", "Downsampling factor", "Datataking Machine", "Telescope", "Number of bits per sample"]
for element in range(len(key_to_delete)):
    delete_info(key_to_delete[element])


os.rename(logfile, str(filfile)+".txt")

