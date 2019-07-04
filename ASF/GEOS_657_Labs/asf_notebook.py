import os  # for chdir, getcwd, path.exists
import re
import time  # for perf_counter
import requests  # for post, get
from getpass import getpass
import json  # for json
import zipfile  # for extractall, ZipFile, BadZipFile
from IPython.display import clear_output
from asf_hyp3 import API  # for get_products, get_subscriptions, login
from getpass import getpass  # used to input URS creds and add to .netrc
import gdal  # for Open
import numpy as np
import datetime
import glob

import subprocess              #only keep if geotiff_from_plot stays in module
import matplotlib.pylab as plt #only keep if geotiff_from_plot stays in module

# path_exists()
# Takes a string path, returns true if exists or
# prints error message and returns false if it doesn't.
def path_exists(path):
    if os.path.exists(path):
        return True
    else:
        print(f"Invalid Path: {path}")
        return False

# new_directory()
# Takes a path for a new or existing directory. Creates directory
# and sub-directories if not already present.


def new_directory(path):
    if os.path.exists(path):
        print(f"{path} already exists.")
    else:
        os.makedirs(path)
        print(f"Created: {path}")
    if not os.path.exists(path):
        print(f"Failed to create path!")


# download()
# Takes a filename and get or post request, then downloads the file
# while outputting a download status bar.
# Preconditions:
# - filename must be valid
def download(filename, request):
    with open(filename, 'wb') as f:
        start = time.perf_counter()
        if request is None:
            f.write(request.content)
        else:
            total_length = int(request.headers.get('content-length'))
            dl = 0
            for chunk in request.iter_content(chunk_size=1024*1024):
                dl += len(chunk)
                if chunk:
                    f.write(chunk)
                    f.flush()
                    done = int(50 * dl / int(total_length))
                    print("\r[%s%s] %s bps, %s%%    " % ('=' * done, ' ' * (50-done), dl//(
                        time.perf_counter() - start), int((100*dl)/total_length)), end='\r', flush=True)


# ASF_unzip()
# Takes a destination directory and file path.
# If file is a valid zip, it extracts all to the destination directory.
# Preconditions:
def ASF_unzip(destination, file_path):
    if path_exists(destination):
        file_name, ext = os.path.splitext(file_path)
        if ext == ".zip":
            print(f"Extracting: {file_path}")
            try:
                zipfile.ZipFile(file_path).extractall(destination)
            except zipfile.BadZipFile:
                print(f"Zipfile Error.")
            return


def remove_nan_subsets(path, tiff_paths):
    if tiff_paths:
        removed = 0
        zero_totals = []
        for tiff in tiff_paths:
            raster = gdal.Open(f"{path}{tiff}")
            if raster:
                band = raster.ReadAsArray()
                zero_count = np.size(band) - np.count_nonzero(band)
                zero_totals.append(zero_count)
        if zero_totals:
            least_zeros = min(zero_totals)
            for i in range(0, len(zero_totals)):
                if zero_totals[i] > int(least_zeros*1.05):
                    os.remove(f"{path}{tiff_paths[i]}")
                    removed += 1
        print(f"GeoTiffs Examined: {len(tiff_paths)}")
        print(f"GeoTiffs Removed:  {removed}")
    else:
        print(f"Error: No tiffs were passed to remove_nan_subsets")

'''
# do not include a file extension in out_filename
# extent must be in the form of a list: [[upper_left_x, upper_left_y], [lower_right_x, lower_right_y]]
def geotiff_from_plot(source_image, out_filename, extent, predominate_utm, cmap=None, vmin=None, vmax=None):
    plt.figure()
    plt.axis('off')
    plt.imshow(source_image, cmap=cmap, vmin=vmin, vmax=vmax)
    temp = f"{out_filename}_temp.png"
    
    print(temp)
    print(f"{out_filename}.tiff")
    
    plt.savefig(temp, dpi=300, transparent='true', bbox_inches='tight', pad_inches=0)

    #cmd = f"gdal_translate -of Gtiff -a_ullr {extent[0][0]} {extent[0][1]} {extent[1][0]} {extent[1][1]} -a_srs EPSG:{predominate_utm} {temp} {out_filename}.tiff"
    
    subprocess.call([f"gdal_translate", f"-of Gtiff", f"-a_ullr {extent[0][0]} {extent[0][1]} {extent[1][0]} {extent[1][1]}", f"-a_srs EPSG:{predominate_utm}", f"{temp}", f"{out_filename}.tiff"]) 
    #!{cmd}
    try:
        os.remove(temp)
    except FileNotFoundError:
        pass
'''        
        
#####################
#  Earth Data Login #
#####################

def earthdata_login():
    print(f"Enter your NASA EarthData username:")
    username = input()
    print(f"Enter your password:")
    password = getpass()

    filename = "/home/jovyan/.netrc"
    with open(filename, 'w+') as f:
        f.write(
            f"machine urs.earthdata.nasa.gov login {username} password {password}\n")

    api = API(username)
    api.login(password)
    return api


#########################
#  Vertex API Functions #
#########################

# get_vertex_granule_info()
# Takes a string granule name and int data level, and returns the granule info as json.<br><br>
# preconditions:
# Requires AWS Vertex API authentification.
# Requires a valid granule name.
# Granule and processing level must match.


def get_vertex_granule_info(granule_name, processing_level):
    vertex_API_URL = "https://api.daac.asf.alaska.edu/services/search/param"
    response = requests.post(
        vertex_API_URL,
        params=[('granule_list', granule_name), ('output', 'json'),
                ('processingLevel', processing_level)]
    )
    if response.status_code == 401:
        pwd = getpass('Password for {}: '.format(username))
        response = requests.post(
            vertex_API_URL,
            params=[('granule_list', granule_name), ('output', 'json'), ],
            stream=True,
            auth=(username, pwd)
        )
    if response.json()[0]:
        json_response = response.json()[0][0]
        return json_response
    else:
        print("get_vertex_granule_info() failed.\ngranule/processing level mismatch.")

        

# download_ASF_granule()
# Takes a string granule name and int data level, then downloads the associated granule and returns its file name.<br><br>
# preconditions:
# Requires AWS Vertex API authentification.
# Requires a valid granule name.
# Granule and processing level must match.


def download_ASF_granule(granule_name, processing_level):
    vertex_info = get_vertex_granule_info(granule_name, processing_level)
    url = vertex_info["downloadUrl"]
    local_filename = vertex_info["fileName"]
    # NOTE stream=True is required for chunking
    r = requests.post(url, stream=True)
    if r.status_code == 401:
        pwd = getpass('Password for {}: '.format(username))
        r = requests.post(r.url, stream=True, auth=(username, pwd))
    total_length = int(r.headers.get('content-length'))
    if os.path.exists(local_filename):
        if os.stat(local_filename).st_size == total_length:
            print(
                f"{local_filename} is already present in current working directory.")
            return local_filename
    print(f"Downloading {url}")
    download(local_filename, r)
    if os.stat(local_filename).st_size < total_length:
        print('\nDownload failed!\n')
        return
    else:
        print('\nDone\n')
        return local_filename


#######################
#  Hyp3 API Functions #
#######################

# get_hyp3_subscriptions
# Takes a Hyp3 API object and returns a list of enabled associated subscriptions
# Returns None if there are no enabled subscriptions associated with Hyp3 account.
# precondition: must already be logged into hyp3


def get_hyp3_subscriptions(hyp3_api_object):
    subscriptions = hyp3_api_object.get_subscriptions(enabled=True)
    if not subscriptions:
        print("There are no subscriptions associated with this Hyp3 account.")
    return subscriptions



# pick_hyp3_subscription
# Takes a list of Hyp3 subscriptions, prompts the user to pick a subcription ID number, and returns that ID number.
# Returns None if subscription list is empty


def pick_hyp3_subscription(subscriptions):
    if subscriptions:
        subscription_ids = []
        while(True):
            subscription_id = None
            while not subscription_id:
                print(f"Enter a subscription ID number:")
                for subscription in subscriptions:
                    print(
                        f"\nSubscription id: {subscription['id']} {subscription['name']}")
                    subscription_ids.append(subscription['id'])
                try:
                    subscription_id = int(input())
                except ValueError:
                    clear_output()
                    print("Invalid ID\nPick a subscription ID from the above list.")
            if subscription_id in subscription_ids:
                break
            else:
                print("Invalid ID\nPick a valid subscription ID from the list.\n")
                clear_output()
        return subscription_id


                        
# download_hyp3_products()
# Takes a Hyp3 API object and a destination path.
# Calls pick_hyp3_subscription() and downloads all products associated with the selected subscription. Returns subscription id.
# preconditions:# -must already be logged into hyp3
#                 -path must be valid
                        
                        
def download_hyp3_products_v2(hyp3_api_object, path, count):
    subscriptions = get_hyp3_subscriptions(hyp3_api_object)
    subscription_id = pick_hyp3_subscription(subscriptions)
    if subscription_id:
        products = []
        page_count = 0
        while True:
            product_page = hyp3_api_object.get_products(
                sub_id=subscription_id, page=page_count, page_size=100)
            page_count += 1
            if not product_page:
                break
            for product in product_page:
                products.append(product)
        if path_exists(path):
            print(
                f"\n{len(products)} product/s associated with Subscription ID: {subscription_id}\n")
            for p in range(0, count):
                url = products[p]['url']
                _match = re.match(
                    r'https://hyp3-download.asf.alaska.edu/asf/data/(.*).zip', url)
                product = _match.group(1)
                filename = f"{path}/{product}"
                # if not already present, we need to download and unzip products
                if not os.path.exists(filename):
                    print(f"\n{product} is not present.\nDownloading from {url}")
                    r = requests.get(url, stream=True)
                    download(filename, r)
                    print(f"\n")
                    os.rename(filename, f"{filename}.zip")
                    filename = f"{filename}.zip"
                    ASF_unzip(path, filename)
                    os.remove(filename)
                    print(f"\nDone.")
                else:
                    print(f"{filename} already exists.")
        return subscription_id

                        
def polarization_exists(paths):
    pth = glob.glob(paths)
    if pth:
        return True
    else:
        return False                        
                        
def select_RTC_polarization(process_type, base_path):
    polarizations = []
    if process_type == 2: # Gamma
        if polarization_exists(f"{base_path}/*/*_VV.tif"):
            polarizations.append('_VV')
        if polarization_exists(f"{base_path}/*/*_VH.tif"):
            polarizations.append('_VH')
        if polarization_exists(f"{base_path}/*/*_HV.tif"):
            polarizations.append('_HV')
        if polarization_exists(f"{base_path}/*/*_HH.tif"):
            polarizations.append('_HH')
    elif process_type == 18: # S1TBX
        if polarization_exists(f"{base_path}/*/*-VV.tif"):
            polarizations.append('-VV')
        if polarization_exists(f"{base_path}/*/*-VH.tif"):
            polarizations.append('-VH')
        if polarization_exists(f"{base_path}/*/*-HV.tif"):
            polarizations.append('-HV')
        if polarization_exists(f"{base_path}/*/*-HH.tif"):
            polarizations.append('-HH')
    if len(polarizations) == 1:
        print(f"Selecting the only available polarization: {polarizations[0]}")
        return f"{base_path}/*/*{polarizations[0]}.tif"
    elif len(polarizations) > 1:
        print(f"Select a polarization:")
        for i in range(0,len(polarizations)):
            print(f"[{i}]: {polarizations[i]}")
        while(True):
            user_input = input()
            try:
                choice = int(user_input)
            except ValueError:
                print(f"Please enter the number of an available polarization.")
                continue
            if choice > len(polarizations) or choice < 0:
                print(f"Please enter the number of an available polarization.")
                continue               
            return f"{base_path}/*/*{polarizations[choice]}.tif"
    else:
        print(f"Error: found no available polarizations.")                       
                        
# date_range_valid()
# Takes a start and end date. 
# Returns True if start_date <= end_date, else prints an error message and returns False.
# Preconditions: start_date and end_date must be datetime.date objects
                        
                        
def date_range_valid(start_date, end_date):
    if start_date and end_date:
        if start_date > end_date:
            print("Error: The start date must be prior to the end date.")
        else:
            return True
    elif (start_date and not end_date) or (not start_date and end_date):
        if not start_date:
            print("Error: An end date was passed, but not a start date.")
        else:
            print("Error: A start date was passed, but not an end date.")
        return False
    else:                
        return True

                        
                        
# get_aquisition_date_from_product_name()
# Takes a json dict containing the product name under the key 'name'
# Returns its aquisition date.                        
# Preconditions: product_info must be a dictionary containing product info, as returned from the
#                hyp3_API get_products() function.
                        
                        
def get_aquisition_date_from_product_name(product_info):
    product_name = product_info['name']
    split_name = product_name.split('_')
    if len(split_name) == 1:
        split_name = product_name.split('-')
        d = split_name[1]
        return datetime.date(int(d[0:4]), int(d[4:6]), int(d[6:8]))
    else:                    
        d = split_name[4]
        return datetime.date(int(d[0:4]), int(d[4:6]), int(d[6:8]))

                        

# filter_date_range() 
# Takes a product list and date range.
# Returns filtered list of products falling inside date range.
# Preconditions: - product_list must be a list of dictionaries containing product info, as returned from the
#                  hyp3_API get_products() function.
#                - start_date and end_date must be datetime.date objects
                        
                        
def filter_date_range(product_list, start_date, end_date):
    filtered_products = []                    
    for product in product_list:
        date = get_aquisition_date_from_product_name(product)
        if date >= start_date and date < end_date:
            filtered_products.append(product)
    return filtered_products


                        
# flight_direction_valid()
# Takes a flight direction (or None)
# Returns False if flight direction is not a valid Vertex API flight direction key value
# else returns True
                        
                        
def flight_direction_valid(flight_direction=None):
    if flight_direction:
        valid_directions = ['A', 'ASC', 'ASCENDING', 'D', 'DESC', 'DESCENDING']
        if flight_direction not in valid_directions:
            print(f"Error: {flight_direction} is not a valid flight direction.")
            print(f"Valid Directions: {valid_directions}")           
            return False
    return True
 
                        
                        
# product_filter()
# Takes a list of products, flight_direction(optional) and path(optional)
# Returns a list of products filtered by flight_direction and/or path
                        
                        
def product_filter(product_list, flight_direction=None, path=None):
    filtered_products = []                        
    for product in product_list:                 
        granule_name = product['name']
        
    
        #print(product) ################################################3
        #print(granule_name)##############################################                
                        
        granule_name = granule_name.split('-')[0]
        vertex_API_URL = "https://api.daac.asf.alaska.edu/services/search/param"
        parameters = [('granule_list', granule_name), ('output', 'json')]
        if flight_direction:
            parameters.append(('flightDirection', flight_direction))
        if path:
            parameters.append(('relativeOrbit', path))
        response = requests.post(
            vertex_API_URL,
            params=parameters,
            stream=True
        )
        if response.status_code == 401:
            pwd = getpass('Password for {}: '.format(username))
            response = requests.post(
                vertex_API_URL,
                params=parameters,
                stream=True,
                auth=(username, pwd)
            )               
        json_response = None
        if response.json()[0]:
            json_response = response.json()[0][0]
        if json_response:           
            filtered_products.append(product)  
    return filtered_products  

       
                        
# download_hyp3_products()
# Takes a Hyp3 API object and a destination path.
# Calls pick_hyp3_subscription() and downloads all products associated with the selected subscription. Returns subscription id.
# preconditions: -must already be logged into hyp3
#                -destination_path must be valid


def download_hyp3_products(hyp3_api_object, destination_path, start_date=None, end_date=None, flight_direction=None, path=None):
    subscriptions = get_hyp3_subscriptions(hyp3_api_object)
    subscription_id = pick_hyp3_subscription(subscriptions)
    if subscription_id:
        products = []
        page_count = 0
        product_count = 1
        while True:
            product_page = hyp3_api_object.get_products(
                sub_id=subscription_id, page=page_count, page_size=100)
            page_count += 1
            if not product_page:
                break
            for product in product_page:
                products.append(product)
        if date_range_valid(start_date, end_date) and flight_direction_valid(flight_direction):
            if start_date:
                products = filter_date_range(products, start_date, end_date)
            if flight_direction:
                products = product_filter(products, flight_direction=flight_direction)
            if path:
                products = product_filter(products, path=path) 
        else:         
            return
        if path_exists(destination_path):
            print(f"\n{len(products)} products are associated with the selected date range, flight direction, and path for Subscription ID: {subscription_id}")
            for p in products:
                print(f"\nProduct Number {product_count}:")
                product_count += 1
                url = p['url']
                _match = re.match(
                    r'https://hyp3-download.asf.alaska.edu/asf/data/(.*).zip', url)
                product = _match.group(1)
                filename = f"{destination_path}/{product}"
                # if not already present, we need to download and unzip products
                if not os.path.exists(filename):
                    print(
                        f"\n{product} is not present.\nDownloading from {url}")
                    r = requests.get(url, stream=True)
                    download(filename, r)
                    print(f"\n")
                    os.rename(filename, f"{filename}.zip")
                    filename = f"{filename}.zip"
                    ASF_unzip(destination_path, filename)
                    os.remove(filename)
                    print(f"\nDone.")
                else:
                    print(f"{filename} already exists.")
        return subscription_id

