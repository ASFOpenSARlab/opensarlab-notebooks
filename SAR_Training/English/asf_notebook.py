# asf_notebook.py
# Alex Lewandowski
# 7-28-19
# Module of Alaska Satellite Facility OpenSARLab Jupyter Notebook helper functions 


import os  # for chdir, getcwd, path.exists
import re
import time  # for perf_counter
import requests  # for post, get
from getpass import getpass  # used to input URS creds and add to .netrc
import zipfile  # for extractall, ZipFile, BadZipFile
import datetime
import glob
import sys

import gdal  # for Open
import numpy as np

from IPython.utils.text import SList
from IPython.display import clear_output

from asf_hyp3 import API, LoginError  # for get_products, get_subscriptions, login


#######################
#  Utility Functions  #
#######################


def path_exists(path: str) -> bool:
    """
    Takes a string path, returns true if exists or
    prints error message and returns false if it doesn't.
    """
    assert type(path) == str, 'Error: path must be a string'

    if os.path.exists(path):
        return True
    else:
        print(f"Invalid Path: {path}")
        return False

    
def new_directory(path: str):
    """
    Takes a path for a new or existing directory. Creates directory
    and sub-directories if not already present.
    """
    assert type(path) == str
    
    if os.path.exists(path):
        print(f"{path} already exists.")
    else:
        os.makedirs(path)
        print(f"Created: {path}")
    if not os.path.exists(path):
        print(f"Failed to create path!")



def download(filename: str, request: requests.models.Response):
    """
    Takes a filename and get or post request, then downloads the file
    while outputting a download status bar.
    Preconditions: filename must be valid
    """
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
                    stars = '=' * done
                    spaces = ' ' * (50-done)
                    bps = dl//(time.perf_counter() - start)
                    percent = int((100*dl)/total_length)
                    print(f"\r[{stars}{spaces}] {bps} bps, {percent}%    ", end='\r', flush=True)
                

def asf_unzip(output_dir: str, file_path: str):
    """
    Takes an output directory path and a file path to a zipped archive.
    If file is a valid zip, it extracts all to the output directory.
    """
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
        
    
def remove_nan_filled_tifs(tif_dir: str, file_names: SList):
    """
    Takes a path to a directory containing tifs and
    and a list of the tif filenames.
    Deletes any tifs containing only NaN values.  
    """
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


def earthdata_hyp3_login():
    """
    takes user input to login to NASA Earthdata
    updates .netrc with user credentials
    returns an api object
    note: Earthdata's EULA applies when accessing ASF APIs
          Hyp3 API handles HTTPError and LoginError
    """
    err = None
    while True:
        if err: # Jupyter input handling requires printing login error here to maintain correct order of output.
            print(err)
            print("Please Try again.\n")
        print(f"Enter your NASA EarthData username:")
        username = input()
        print(f"Enter your password:")
        password = getpass()
        try:
            api = API(username) # asf_hyp3 function
        except Exception:
            raise
        else:
            try: 
                api.login(password)
            except LoginError as e:
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


def get_vertex_granule_info(granule_name: str, processing_level: int) -> dict:
    """
    Takes a string granule name and int processing level, and returns the granule info as json.<br><br>
    preconditions:
    Requires AWS Vertex API authentification (already logged in).
    Requires a valid granule name.
    Granule and processing level must match.
    """
    assert type(granule_name) == str, 'Error: granule_name must be a string.'
    assert type(processing_level) == str, 'Error: processing_level must be a string.'

    vertex_API_URL = "https://api.daac.asf.alaska.edu/services/search/param"
    try: 
        response = requests.post(
            vertex_API_URL,
            params=[('granule_list', granule_name), ('output', 'json'),
                    ('processingLevel', processing_level)]
        )
    except requests.exceptions.RequestException as e:  # This is the correct syntax
        print(e)
        sys.exit(1)
    else:
        if len(response.json()) > 0:
            json_response = response.json()[0][0]
            return json_response
        else:
            print("get_vertex_granule_info() failed.\ngranule/processing level mismatch.")
        


def download_ASF_granule(granule_name: str, processing_level: str) -> str:
    """
    Takes a string granule name and string processing level, then downloads the associated granule 
    and returns its file name.<br><br>
    preconditions:
    Requires AWS Vertex API authentification (already logged in).
    Requires a valid granule name.
    Granule and processing level must match.
    """
    assert type(granule_name) == str, 'Error: granule_name must be a string.'
    assert type(processing_level) == str, 'Error: processing_level must be a string.'

    vertex_info = get_vertex_granule_info(granule_name, processing_level)
    url = vertex_info["downloadUrl"]
    local_filename = vertex_info["fileName"]
    try:
        r = requests.post(url, stream=True)
    except requests.exceptions.RequestException as e:
        print(e)
        sys.exit(1)
    else:
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


def get_hyp3_subscriptions(hyp3_api_object: API) -> dict:
    """
    Takes a Hyp3 API object and returns a list of enabled associated subscriptions
    Returns None if there are no enabled subscriptions associated with Hyp3 account.
    precondition: must already be logged into hyp3
    """
    assert type(hyp3_api_object) == API, f"Error: get_hyp3_subscriptions was passed a {type(hyp3_api_object)}, not a asf_hyp3.API object"
    try:
        subscriptions = hyp3_api_object.get_subscriptions(enabled=True)
    except Exception:
        raise
    else:
        if not subscriptions:
            print("There are no subscriptions associated with this Hyp3 account.")
        return subscriptions



def pick_hyp3_subscription(subscriptions: list) -> int:
    """
    Takes a list of Hyp3 subscriptions, prompts the user to pick a subcription ID number, 
    and returns that ID number.
    Returns None if subscription list is empty
    """
    assert type(subscriptions) == list, 'Error: subscriptions must be a list'
    assert len(subscriptions) > 0, 'Error: There are no subscriptions in the passed list'
    
    possible_ids = []
    for subscription in subscriptions:
        print(
            f"\nSubscription id: {subscription['id']} {subscription['name']}")
        possible_ids.append(subscription['id'])
    while True:
        print(f"Pick a subscription ID from the above list:")
        try:
            user_choice = int(input())
            if user_choice in possible_ids:
                return user_choice
        except ValueError:
            print("\nInvalid ID")
        else:
            print("\nInvalid ID")

                    
def polarization_exists(paths: str):
    """
    Takes a wildcard path to images with a particular polarization
    ie. "rtc_products/*/*_VV.tif"
    returns true if any matching paths are found, else false
    """
    assert type(paths) == str, 'Error: must pass string wildcard path of form "rtc_products/*/*_VV.tif"'

    pth = glob.glob(paths)
    if pth:
        return True
    else:
        return False                             
            
                    
def select_RTC_polarization(process_type: int, base_path: str) -> str:
    """
    Takes an int process type and a string path to a base directory
    If files in multiple polarizations found, promts user for a choice.
    Returns string wildcard path to files of selected (or only available)
    polarization
    """
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
        for i in range(0, len(polarizations)):
            print(f"[{i}]: {polarizations[i]}")
        while True:
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

                    
def date_range_valid(start_date: datetime.date = None, end_date: datetime.date = None) -> bool:
    """
    Takes a start and end date. 
    Returns True if start_date <= end_date, else prints an error message and returns False.
    """
    if start_date:
        assert type(start_date) == datetime.date, 'Error: start_date must be a datetime.date'
    if end_date:
        assert type(end_date) == datetime.date, 'Error:, end_date must be a datetime.date'
            
    if start_date is None or end_date is None:
        return False
    elif start_date > end_date:
        print("Error: The start date must be prior to the end date.")
    else:                
        return True
                                              
                        
def get_aquisition_date_from_product_name(product_info: dict) -> datetime.date:
    """
    Takes a json dict containing the product name under the key 'name'
    Returns its aquisition date.                        
    Preconditions: product_info must be a dictionary containing product info, as returned from the
                   hyp3_API get_products() function.
    """
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
                        
                        
def filter_date_range(product_list: list, start_date: datetime.date, end_date: datetime.date) -> list:
    """
    Takes a product list and date range.
    Returns filtered list of product info dictionaries falling inside date range.
    Preconditions: - product_list must be a list of dictionaries containing product info, as returned from the
                     hyp3_API get_products() function.
                   - start_date and end_date must be datetime.date objects
    """
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
                       
                        
def flight_direction_valid(flight_direction: str = None) -> bool:
    """
    Takes a flight direction (or None)
    Returns False if flight direction is not a valid Vertex API flight direction key value
    else returns True
    """
    assert type(flight_direction) == str, 'Error: flight_direction must be a string.'
    valid_directions = ['A', 'ASC', 'ASCENDING', 'D', 'DESC', 'DESCENDING']
    if flight_direction not in valid_directions:
        print(f"Error: {flight_direction} is not a valid flight direction.")
        print(f"Valid Directions: {valid_directions}")           
        return False
    else:
        return True

                        
def product_filter(product_list: list, flight_direction: str = None, path: int = None) -> list:
    """
    Takes a list of products info dictionaries, string flight_direction(optional) and int path(optional)
    Returns a list of products info dictionaries filtered by flight_direction and/or path
    """
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
            try:
                response = requests.post(
                    vertex_API_URL,
                    params=parameters,
                    stream=True
                )
            except requests.exceptions.RequestException as e:
                print(e)
                sys.exit(1)               
            json_response = None
            if response.json()[0]:
                json_response = response.json()[0][0]
            if json_response:           
                filtered_products.append(product)  
        return filtered_products  


def download_hyp3_products(hyp3_api_object: API, 
                           destination_path: str, 
                           start_date: datetime.date = None, 
                           end_date: datetime.date = None, 
                           flight_direction: str = None, 
                           path: int = None) -> int:
    """
    Takes a Hyp3 API object and a destination path.
    Calls pick_hyp3_subscription() and downloads all products associated with the selected subscription. 
    Returns subscription id.
    preconditions: -must already be logged into hyp3
                   -destination_path must be valid
    """
    assert type(hyp3_api_object) == API, 'Error: hyp3_api_object must be an asf_hyp3.API object.'
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
        if date_range_valid(start_date, end_date): 
            products = filter_date_range(products, start_date, end_date)
        if flight_direction: # must check this because both None and incorrect flight_directions 
                             # will return False and it shouldn't exit if flight_direction is None
            if flight_direction_valid(flight_direction):
                products = product_filter(products, flight_direction=flight_direction)
            else:
                print('Aborting download_hyp3_products() due to invalid flight_direction.')
                sys.exit(1)
        if path:
            products = product_filter(products, path=path) 
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