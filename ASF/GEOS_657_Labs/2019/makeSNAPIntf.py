#!/usr/bin/env python


import sys
import numpy as np
import saa_func_lib as saa


i = sys.argv[1]
q = sys.argv[2]

(x,y,trans,proj,idata) = saa.read_gdal_file(saa.open_gdal_file(i))
(x,y,trans,proj,qdata) = saa.read_gdal_file(saa.open_gdal_file(q))


amp = np.sqrt(np.sqrt(np.power(idata,2) + np.power(qdata,2)))
phase = np.arctan2(qdata,idata)


saa.write_gdal_file_float('amplitude.tif',trans,proj,amp)
saa.write_gdal_file_float('phase.tif',trans,proj,phase)









