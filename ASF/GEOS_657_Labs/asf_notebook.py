import os # for chdir, getcwd, path.exists 
import re
import time # for perf_counter
import requests # for post, get
from getpass import getpass
import json # for json
import zipfile # for extractall, ZipFile, BadZipFile
from IPython.display import clear_output
from asf_hyp3 import API # for get_products, get_subscriptions, login
from getpass import getpass # used to input URS creds and add to .netrc
import gdal # for Open
import numpy as np


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
                    print("\r[%s%s] %s bps, %s%%    " % ('=' * done, ' ' * (50-done), dl//(time.perf_counter() - start), int((100*dl)/total_length)), end='\r', flush=True)    
    

# ASF_unzip()
# Takes a destination directory and file path.
# If file is a valid zip, it extracts all to the destination directory.
# Preconditions:
def ASF_unzip(directory_path, file_path):
    if path_exists(directory_path):
        file_name, ext = os.path.splitext(file_path)
        if ext == ".zip":
            print(f"Extracting: {file_path}")
            try:
                zipfile.ZipFile(file_path).extractall(directory_path)
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
        least_zeros = min(zero_totals)
        for i in range (0, len(zero_totals)):
            if zero_totals[i] > least_zeros + int(least_zeros*0.05):
                os.remove(f"{path}{tiff_paths[i]}")
                removed += 1
        print(f"GeoTiffs Examined: {len(tiff_paths)}")
        print(f"GeoTiffs Removed:  {removed}")    
    else:
        print(f"Error: No tiffs were passed to remove_nan_subsets")

        

#####################
#  Earth Data Login #
#####################

def earthdata_login():
    print(f"Enter your NASA EarthData username:")
    username = input()
    print(f"Enter your password:")
    password = getpass()
    
    filename="/home/jovyan/.netrc"
    with open(filename, 'w+') as f:
        f.write(f"machine urs.earthdata.nasa.gov login {username} password {password}\n")
    
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
        params = [('granule_list', granule_name), ('output', 'json'), ('processingLevel', processing_level)]
    )
    if response.status_code == 401:
        pwd = getpass('Password for {}: '.format(username))
        response = requests.post(
            vertex_API_URL, 
            params = [('granule_list', granule_name), ('output', 'json'), ], 
            stream=True, 
            auth=(username,pwd)
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
    r = requests.post(url, stream=True)   # NOTE stream=True is required for chunking
    if r.status_code == 401:
        pwd = getpass('Password for {}: '.format(username))
        r = requests.post(r.url, stream=True, auth=(username,pwd))
    total_length = int(r.headers.get('content-length'))
    if os.path.exists(local_filename):
        if os.stat(local_filename).st_size == total_length:
            print(f"{local_filename} is already present in current working directory.")
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
                    print(f"\nSubscription id: {subscription['id']} {subscription['name']}")
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
# preconditions: 
# -must already be logged into hyp3
# -path must be valid                          
def download_hyp3_products_v2(hyp3_api_object, path, count):
    subscriptions = get_hyp3_subscriptions(hyp3_api_object)
    subscription_id = pick_hyp3_subscription(subscriptions)
    if subscription_id:
        products = []
        page_count = 0
        while True:
            product_page = hyp3_api_object.get_products(sub_id=subscription_id, page=page_count, page_size=100)            
            page_count += 1
            if not product_page:
                break
            for product in product_page:
                products.append(product) 
        if path_exists(path):             
            print(f"\n{len(products)} product/s associated with Subscription ID: {subscription_id}\n")
            for p in range (0, count): # 
                url = products[p]['url']
                _match = re.match(r'https://hyp3-download.asf.alaska.edu/asf/data/(.*).zip', url)
                product = _match.group(1)
                filename = f"{path}/{product}"
                if not os.path.exists(filename): # if not already present, we need to download and unzip products
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
                          
                                         
                          
# download_hyp3_products()
# Takes a Hyp3 API object and a destination path.
# Calls pick_hyp3_subscription() and downloads all products associated with the selected subscription. Returns subscription id.                        
# preconditions: 
# -must already be logged into hyp3
# -path must be valid                          
def download_hyp3_products(hyp3_api_object, path):
    subscriptions = get_hyp3_subscriptions(hyp3_api_object)
    subscription_id = pick_hyp3_subscription(subscriptions)
    if subscription_id:
        products = []
        page_count = 0
        product_count = 1
        while True:
            product_page = hyp3_api_object.get_products(sub_id=subscription_id, page=page_count, page_size=100)            
            page_count += 1
            if not product_page:
                break
            for product in product_page:
                products.append(product)
        if path_exists(path):             
            print(f"\n{len(products)} product/s associated with Subscription ID: {subscription_id}")
            for p in products:
                print(f"\nProduct Number {product_count}:")
                product_count += 1
                url = p['url']
                _match = re.match(r'https://hyp3-download.asf.alaska.edu/asf/data/(.*).zip', url)
                product = _match.group(1)
                filename = f"{path}/{product}"
                if not os.path.exists(filename): # if not already present, we need to download and unzip products
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
                          
                          
