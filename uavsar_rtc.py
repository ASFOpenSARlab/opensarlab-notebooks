# UAVSAR - NISAR Simulated
# Original code by Josef Kellndorfer
# Revised by Alan Xu (08/21/2020)

import os
from math import pi
from osgeo import gdal
import numpy as np
from osgeo import osr
import pandas as pd


def get_product_ids(datatakes="datatakes"):
    df = pd.read_csv(datatakes)
    dt_tmp = df.iloc[:, 0].tolist()
    dt = {}
    for i in dt_tmp:
        tokens = i[3:].split('_')
        dtroot = '_'.join(tokens[:-1])
        if not dtroot in dt:
            dt[dtroot] = []
        dt[dtroot].append(tokens[-1])
    print(dt)

    product_ids = []
    for i in dt:
        vers = dt[i][-1]
        if vers == '04':
            vers = dt[i][-2]
        product_ids.append(i + '_' + vers)
    product_ids.sort()
    print('\n'.join(product_ids))

    return product_ids


def get_download_links(product_ids, root='http://downloaduav.jpl.nasa.gov/Release2u/'):
    dl = {}
    for product_id in product_ids:
        dl0 = {'ann': root + product_id + '/' + product_id.replace('_CX_', '_CX_129A_') + '.ann',
               'rtcf': root + product_id + '/' + product_id.replace('_CX_', '_CX_129A_') + '.rtc',
               'inc': root + product_id + '/' + product_id.replace('_CX_',
                                                                   '_CX_129A_') + '.inc',
               'flatinc': root + product_id + '/' + product_id.replace('_CX_',
                                                                       '_CX_129A_') + '.flat.inc',
               'hh': root + product_id + '/' + product_id.replace('_CX_', 'HHHH_CX_129A_') + '.grd',
               'hv': root + product_id + '/' + product_id.replace('_CX_', 'HVHV_CX_129A_') + '.grd'}
        dl[product_id] = dl0
    return dl


def get_dithered_download_links(product_ids, root='http://downloaduav.jpl.nasa.gov/Release2u/'):
    dl = {}
    for product_id in product_ids:
        dl0 = {'ann': root + product_id + '/' + product_id.replace('_CX_', '_CD_129_') + '.ann',
               'h5': root + product_id + '/' + product_id.replace('_CX_', '_CD_129_') + '.h5'}
        dl[product_id] = dl0
    return dl


def ann2dict(ann):
    # Make a dictionary from the annotation file
    with open(ann, "r") as f:
        lines = f.readlines()
    # Skip comments and empty lines
    lines = [x for x in lines if (not x.startswith(';') and (not x.startswith('\n')))]
    validline = False
    anndict = {}
    for l in lines:
        if l.startswith('slc_mag.set_rows'):
            validline = True
        if not validline or l.startswith('comments'):
            next
        else:
            key = l.split()[0]
            units = l.split('(')[1].split(')')[0]
            value = l.split('=')[1].split()[0].strip(' ')
            anndict[key] = {'units': units, 'value': value}
    return anndict


def grd2tif(product_id, dl, overwrite=True):
    # Make the annotation dictionary
    ann = os.path.basename(dl[product_id]['ann'])
    anndict = ann2dict(ann)

    # Get the row/cols
    cols = int(anndict['grd_pwr.set_cols']['value'])
    rows = int(anndict['grd_pwr.set_rows']['value'])

    # Read the grds and rtcf into numpy arrays
    ## Get the local filenames
    hh_name = os.path.basename(dl[product_id]['hh'])
    hv_name = os.path.basename(dl[product_id]['hv'])
    rtcf_name = os.path.basename(dl[product_id]['rtcf'])
    inc_name = os.path.basename(dl[product_id]['inc'])
    flatinc_name = os.path.basename(dl[product_id]['flatinc'])
    ## Read into numpy arrays and reshape to row/cols dimensions
    hh = np.fromfile(hh_name, dtype=np.float32).reshape((rows, cols))
    hv = np.fromfile(hv_name, dtype=np.float32).reshape((rows, cols))
    rtcf = np.fromfile(rtcf_name, dtype=np.float32).reshape((rows, cols))
    inc = np.fromfile(inc_name, dtype=np.float32).reshape((rows, cols))
    flatinc = np.fromfile(flatinc_name, dtype=np.float32).reshape((rows, cols))

    # Mask for NISAR Inc. Angle Range
    ## Build Mask
    nisar_near_range_inc_deg = 33
    nisar_far_range_inc_deg = 47
    nisar_near_range_inc_rad = nisar_near_range_inc_deg / 180 * pi
    nisar_far_range_inc_rad = nisar_far_range_inc_deg / 180 * pi
    mask_nisar_inc = ~((flatinc >= nisar_near_range_inc_rad) & (flatinc <= nisar_far_range_inc_rad))
    ## Apply mask to grd and rtcf arrays
    hh_masked = np.ma.array(hh, mask=mask_nisar_inc, fill_value=np.nan)
    hv_masked = np.ma.array(hv, mask=mask_nisar_inc, fill_value=np.nan)
    rtcf_masked = np.ma.array(rtcf, mask=mask_nisar_inc, fill_value=np.nan)
    inc_masked = np.ma.array(inc, mask=mask_nisar_inc, fill_value=np.nan)

    # Apply rtcf to grd to generate rtc products
    hh_rtc = hh_masked * rtcf_masked / np.cos(inc_masked)
    hv_rtc = hv_masked * rtcf_masked / np.cos(inc_masked)

    drv = gdal.GetDriverByName('GTiff')

    # Set the geotransformation tuple
    xres = float(anndict['grd_pwr.col_mult']['value'])
    yres = float(anndict['grd_pwr.row_mult']['value'])
    # set upper left pixel as pixel as Area
    ulx = float(anndict['grd_pwr.col_addr']['value']) - xres / 2
    uly = float(anndict['grd_pwr.row_addr']['value']) - yres / 2  # Note that yres is negative
    geotrans = (ulx, xres, 0, uly, 0, yres)

    # Projection info for Lon/lat (EPSG 4326)
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS("WGS84")
    proj = srs.ExportToWkt()

    tifnames = []
    for tifname, arr in zip((hh_name, hv_name), (hh_rtc, hv_rtc)):
        tifname = tifname.replace('.grd', '.tif')
        # Create Image with geocoding info
        make_image = True
        if os.path.exists(tifname):
            if overwrite:
                os.remove(tifname)
            else:
                make_image = False
                print('Not overwriting existing', tifname)
        if make_image:
            img = drv.Create(tifname, cols, rows, 1, gdal.GDT_Float32)
            img.SetGeoTransform(geotrans)
            img.SetProjection(proj)
            # Write data to the band
            band = img.GetRasterBand(1)
            # Write raster data
            band.WriteArray(arr.filled())
            # Set no data valye
            band.SetNoDataValue(np.nan)
            # Close the band and image
            band = None
            img = None
            print('Wrote', tifname)

        tifnames.append(tifname)
    drv = None
    return tifnames


def grd2tif0(product_id, dl, overwrite=True):
    # Make the annotation dictionary
    ann = os.path.basename(dl[product_id]['ann'])
    anndict = ann2dict(ann)

    # Get the row/cols
    cols = int(anndict['grd_pwr.set_cols']['value'])
    rows = int(anndict['grd_pwr.set_rows']['value'])

    # Read the grds and rtcf into numpy arrays
    ## Get the local filenames
    hh_name = os.path.basename(dl[product_id]['hh'])
    hv_name = os.path.basename(dl[product_id]['hv'])
    rtcf_name = os.path.basename(dl[product_id]['rtcf'])
    inc_name = os.path.basename(dl[product_id]['inc'])
    flatinc_name = os.path.basename(dl[product_id]['flatinc'])
    ## Read into numpy arrays and reshape to row/cols dimensions
    hh = np.fromfile(hh_name, dtype=np.float32).reshape((rows, cols))
    hv = np.fromfile(hv_name, dtype=np.float32).reshape((rows, cols))
    rtcf = np.fromfile(rtcf_name, dtype=np.float32).reshape((rows, cols))
    inc = np.fromfile(inc_name, dtype=np.float32).reshape((rows, cols))
    flatinc = np.fromfile(flatinc_name, dtype=np.float32).reshape((rows, cols))

    # Apply rtcf to grd to generate rtc products
    hh_rtc = hh * rtcf / np.cos(inc)
    hv_rtc = hv * rtcf / np.cos(inc)

    drv = gdal.GetDriverByName('GTiff')

    # Set the geotransformation tuple
    xres = float(anndict['grd_pwr.col_mult']['value'])
    yres = float(anndict['grd_pwr.row_mult']['value'])
    # set upper left pixel as pixel as Area
    ulx = float(anndict['grd_pwr.col_addr']['value']) - xres / 2
    uly = float(anndict['grd_pwr.row_addr']['value']) - yres / 2  # Note that yres is negative
    geotrans = (ulx, xres, 0, uly, 0, yres)

    # Projection info for Lon/lat (EPSG 4326)
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS("WGS84")
    proj = srs.ExportToWkt()

    tifnames = []
    for tifname, arr in zip((hh_name, hv_name), (hh_rtc, hv_rtc)):
        tifname = tifname.replace('.grd', '.tif')
        # Create Image with geocoding info
        make_image = True
        if os.path.exists(tifname):
            if overwrite:
                os.remove(tifname)
            else:
                make_image = False
                print('Not overwriting existing', tifname)
        if make_image:
            img = drv.Create(tifname, cols, rows, 1, gdal.GDT_Float32)
            img.SetGeoTransform(geotrans)
            img.SetProjection(proj)
            # Write data to the band
            band = img.GetRasterBand(1)
            # Write raster data
            band.WriteArray(arr)
            # Set no data valye
            band.SetNoDataValue(np.nan)
            # Close the band and image
            band = None
            img = None
            print('Wrote', tifname)

        tifnames.append(tifname)
    drv = None
    return tifnames
