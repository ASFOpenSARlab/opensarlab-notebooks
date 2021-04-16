#!/usr/bin/env python


import os,glob
import numpy as np
import saa_func_lib as saa

def gdal_interferogram(folder_in=None, name=''):
    if folder_in is None:
        folder_in = os.getcwd()

    qfile = glob.glob(os.path.join(folder_in, 'q_*.img'))[0]
    ifile = qfile.replace('q_', 'i_')
    (x,y,trans,proj,idata) = saa.read_gdal_file(saa.open_gdal_file(ifile))
    (x,y,trans,proj,qdata) = saa.read_gdal_file(saa.open_gdal_file(qfile))

    #amp = np.sqrt(np.sqrt(np.power(idata,2) + np.power(qdata,2)))
    amp = np.log10((np.sqrt(np.sqrt(np.power(idata,2) + np.power(qdata,2)))+1)/100000000)
    amp = (amp - np.min(amp))
    ma = amp[amp != 0]
    amp = amp - np.min(ma)
    #amp = np.where(amp < 0, amp, 0)
    amp = amp/np.max(amp)*255
    phase = (np.arctan2(qdata,idata)+np.pi)/(2*np.pi)*255


    saa.write_gdal_file_byte(os.path.join(folder_in, f'amplitude{name}.tif'),trans,proj,amp)
    saa.write_gdal_file_byte(os.path.join(folder_in, f'phase{name}.tif'),trans,proj,phase)









