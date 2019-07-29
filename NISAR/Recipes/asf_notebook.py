# asf_notebook.py
# Alex Lewandowski
# 7-28-19
# Module of Alaska Satellite Facility OpenSARLab Jupyter Notebook helper functions 


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
import asf_hyp3
from IPython.utils.text import SList


#######################
#  Utility Functions  #
#######################

# path_exists()
# Takes a string path, returns true if exists or
# prints error message and returns false if it doesn't.


def path_exists(path: str) -> bool:
    assert type(path) == str, 'Error: path must be a string'

    if os.path.exists(path):
        return True
    else:
        print(f"Invalid Path: {path}")
        return False


    
# new_directory()
# Takes a path for a new or existing directory. Creates directory
# and sub-directories if not already present.


def new_directory(path: str):
    assert type(path) == str
    
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
# Preconditions: filename must be valid


def download(filename: str, request: requests.models.Response):
    assert type(filename) == str, 'Error: filename must be a string'
    assert type(request) == requests.models.Response, 'Error: request must be a class<requests.models.Response>'
    
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



# asf_unzip()
# Takes an output directory path and a file path to a zipped archive.
# If file is a valid zip, it extracts all to the output directory.


def asf_unzip(output_dir: str, file_path: str):
    ext = os.path.splitext(file_path)[1]
    assert type(output_dir) == str, 'Error: output_dir must be a string'
    assert type(file_path) == str, 'Error: file_path must be a string'
    assert ext == '.zip', 'Error: file_path must be the path of a zip'

    if path_exists(output_dir):
        if path_exists(file_path):
            print(f"Extracting: {file_path}")
            try:
                zipfile.ZipFile(file_path).extractall(output_dir)
            except zipfile.BadZipFile:
                print(f"Zipfile Error.")
            return



 # remove_nan_filled_tifs()
 # Takes a path to a directory containing tifs and
 # and a list of the tif filenames.
 # Deletes any tifs containing only NaN values.    
    
    
def remove_nan_filled_tifs(tif_dir: str, file_names: SList):
    assert type(tif_dir) == str, 'Error: tif_dir must be a string'
    assert type(file_names) == SList, 'Error: file_names must be an IPython.utils.text.SList'
    assert len(file_names) > 0, 'Error: file_names must contain at least 1 file name'
    
    removed = 0
    for tiff in file_names:
        raster = gdal.Open(f"{tif_dir}{tiff}")
        if raster:
            band = raster.ReadAsArray()
            if np.count_nonzero(band) < 1:
                os.remove(f"{tif_dir}{tiff}")
                removed += 1
    print(f"GeoTiffs Examined: {len(file_names)}")
    print(f"GeoTiffs Removed:  {removed}")


        
########################
#  Earth Data Function #
########################

# eartdata_login()
# takes user input to login to NASA Earthdata
# updates .netrc with user credentials
# returns an api object
# note: Earthdata's EULA applies when accessing ASF APIs
#       Hyp3 API handles HTTPError and LoginError


def earthdata_hyp3_login():
    err = None
    while(True):
        if err: # Jupyter input handling requires printing login error here to maintain correct order of output.
            print(err)
            print("Please Try again.\n")
        print(f"Enter your NASA EarthData username:")
        username = input()
        print(f"Enter your password:")
        password = getpass()
        try:
            api = API(username) # asf_hyp3 function
        except:
            raise
        else:
            try: 
                api.login(password)
            except asf_hyp3.LoginError as e:
                err = e
                clear_output()
                continue
            except Exception:
                    raise
            else:
                clear_output()
                print(f"Login successful.")
                print(f"Welcome {username}.")
                filename = "/home/jovyan/.netrc"
                with open(filename, 'w+') as f:
                    f.write(
                        f"machine urs.earthdata.nasa.gov login {username} password {password}\n")
                return api



#########################
#  Vertex API Functions #
#########################

# get_vertex_granule_info()
# Takes a string granule name and int processing level, and returns the granule info as json.<br><br>
# preconditions:
# Requires AWS Vertex API authentification (already logged in).
# Requires a valid granule name.
# Granule and processing level must match.


def get_vertex_granule_info(granule_name: str, processing_level: int) -> dict:
    assert type(granule_name) == str, 'Error: granule_name must be a string.'
    assert type(processing_level) == str, 'Error: processing_level must be a string.'

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
    if len(response.json()) > 0:
        json_response = response.json()[0][0]
        return json_response
    else:
        print("get_vertex_granule_info() failed.\ngranule/processing level mismatch.")
        


# download_ASF_granule()
# Takes a string granule name and int data level, then downloads the associated granule 
# and returns its file name.<br><br>
# preconditions:
# Requires AWS Vertex API authentification (already logged in).
# Requires a valid granule name.
# Granule and processing level must match.


def download_ASF_granule(granule_name, processing_level) -> str:
    assert type(granule_name) == str, 'Error: granule_name must be a string.'
    assert type(processing_level) == str, 'Error: processing_level must be a string.'

    vertex_info = get_vertex_granule_info(granule_name, processing_level)
    url = vertex_info["downloadUrl"]
    local_filename = vertex_info["fileName"]
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


def get_hyp3_subscriptions(hyp3_api_object: asf_hyp3.API) -> dict:
    assert type(hyp3_api_object) == asf_hyp3.API, f"Error: get_hyp3_subscriptions was passed a {type(hyp3_api_object)}, not a asf_hyp3.API object"
    try:
        subscriptions = hyp3_api_object.get_subscriptions(enabled=True)
    except Exception:
        raise
    else:
        if not subscriptions:
            print("There are no subscriptions associated with this Hyp3 account.")
        return subscriptions



# pick_hyp3_subscription
# Takes a list of Hyp3 subscriptions, prompts the user to pick a subcription ID number, and returns that ID number.
# Returns None if subscription list is empty


def pick_hyp3_subscription(subscriptions: list) -> int:
    assert type(subscriptions) == list, 'Error: subscriptions must be a list'
    assert len(subscriptions) > 0, 'Error: There are no subscriptions in the passed list'
    
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



# polarization_exists()
# Takes a wildcard path to images with a particular polarization
# ie. "rtc_products/*/*_VV.tif"
# returns true if any matching paths are found, else false

                    
def polarization_exists(paths: str):
    assert type(paths) == str, 'Error: must pass string wildcard path of form "rtc_products/*/*_VV.tif"'

    pth = glob.glob(paths)
    if pth:
        return True
    else:
        return False                        



# select_RTC_polarization()
# Takes an int process type and a string path to a base directory
# If files in multiple polarizations found, promts user for a choice.
# Returns string wildcard path to files of selected (or only available)
# polarization     
            
                    
def select_RTC_polarization(process_type: int, base_path: str) -> str:
    assert process_type == 2 or process_type == 18, 'Error: process_type must be 2 (GAMMA) or 18 (S1TBX).'
    assert type(base_path) == str, 'Error: base_path must be a string.'
    assert os.path.exists(base_path), f"Error: select_RTC_polarization was passed an invalid base_path, {base_path}"
    
    polarizations = []
    if process_type == 2: # Gamma
        separator = '_'
    elif process_type == 18: # S1TBX
        separator = '-'                
    if polarization_exists(f"{base_path}/*/*{separator}VV.tif"):
        polarizations.append(f"{separator}VV")
    if polarization_exists(f"{base_path}/*/*{separator}VH.tif"):
        polarizations.append(f"{separator}VH")
    if polarization_exists(f"{base_path}/*/*{separator}HV.tif"):
        polarizations.append(f"{separator}HV")
    if polarization_exists(f"{base_path}/*/*{separator}HH.tif"):
        polarizations.append(f"{separator}HH")  
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
                   
                    
def date_range_valid(start_date: datetime.date, end_date: datetime.date) -> bool:
    assert type(start_date) == datetime.date, 'Error: start_date must be a datetime.date'
    assert type(end_date) == datetime.date, 'Error:, end_date must be a datetime.date'

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
                        
                        
def get_aquisition_date_from_product_name(product_info: dict) -> datetime.date:
    assert type(product_info) == dict, 'Error: product_info must be a dictionary.'
                    
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
# Returns filtered list of product info dictionaries falling inside date range.
# Preconditions: - product_list must be a list of dictionaries containing product info, as returned from the
#                  hyp3_API get_products() function.
#                - start_date and end_date must be datetime.date objects
                        
                        
def filter_date_range(product_list: list, start_date: datetime.date, end_date: datetime.date) -> list:
    assert type(product_list) == list, 'Error: product_list must be a list of product_info dictionaries.'
    assert len(product_list) > 0, 'Error: product_list must contain at least one product_info dictionary.'
    for info_dict in product_list:
        assert type(info_dict) == dict, 'Error: product_list must be a list of product info dictionaries.'
               
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
                        
                        
def flight_direction_valid(flight_direction: str=None) -> bool:
    if flight_direction:
        valid_directions = ['A', 'ASC', 'ASCENDING', 'D', 'DESC', 'DESCENDING']
        if flight_direction not in valid_directions:
            print(f"Error: {flight_direction} is not a valid flight direction.")
            print(f"Valid Directions: {valid_directions}")           
            return False
    return True
 
                        
                        
# product_filter()
# Takes a list of products info dictionaries, string flight_direction(optional) and int path(optional)
# Returns a list of products info dictionaries filtered by flight_direction and/or path
                        
                        
def product_filter(product_list: list, flight_direction: str=None, path: int=None) -> list:
    assert type(product_list) == list, 'Error: product_list must be a list of product_info dictionaries.'
    assert len(product_list) > 0, 'Error: product_list must contain at least one product_info dictionary.'
    for info_dict in product_list:
        assert type(info_dict) == dict, 'Error: product_list must be a list of product info dictionaries.'
    if flight_direction:
        assert type(flight_direction) == str, 'Error: flight_direction must be a string.'
    if path:
        assert type(path) == int, 'Error: path must be an integer.'
                    
    if flight_direction or path:
        filtered_products = []                        
        for product in product_list:                 
            granule_name = product['name']
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


def download_hyp3_products(hyp3_api_object: asf_hyp3.API, 
                           destination_path: str, 
                           start_date: datetime.date=None, 
                           end_date: datetime.date=None, 
                           flight_direction: str=None, 
                           path: int=None) -> int:
    assert type(hyp3_api_object) == asf_hyp3.API, 'Error: hyp3_api_object must be an asf_hyp3.API object.'
    assert type(destination_path) == str, 'Error: destination_path must be a string'
    assert os.path.exists(destination_path), 'Error: desitination_path must be valid'
    if start_date:
        assert type(start_date) == datetime.date, 'Error: start_date must be a datetime.date'
    if end_date:
        assert type(end_date) == datetime.date, 'Error:, end_date must be a datetime.date'
    if flight_direction:
        assert type(flight_direction) == str, 'Error: flight_direction must be a string.'
    if path:
        assert type(path) == int, 'Error: path must be an integer.'
                    
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
                    asf_unzip(destination_path, filename)
                    os.remove(filename)
                    print(f"\nDone.")
                else:
                    print(f"{filename} already exists.")
        return subscription_id