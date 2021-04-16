#!/usr/bin/env python


import sys
import numpy as np
import saa_func_lib as saa


i = sys.argv[1]
q = sys.argv[2]

(x,y,trans,proj,idata) = saa.read_gdal_file(saa.open_gdal_file(i))
(x,y,trans,proj,qdata) = saa.read_gdal_file(saa.open_gdal_file(q))

#amp = np.sqrt(np.sqrt(np.power(idata,2) + np.power(qdata,2)))
amp = np.log10((np.sqrt(np.sqrt(np.power(idata,2) + np.power(qdata,2)))+1)/100000000)
amp = (amp - np.min(amp))
ma = amp[amp != 0]
amp = amp - np.min(ma)
#amp = np.where(amp < 0, amp, 0)
amp = amp/np.max(amp)*255
phase = (np.arctan2(qdata,idata)+np.pi)/(2*np.pi)*255


saa.write_gdal_file_byte('amplitude.tif',trans,proj,amp)
saa.write_gdal_file_byte('phase.tif',trans,proj,phase)









