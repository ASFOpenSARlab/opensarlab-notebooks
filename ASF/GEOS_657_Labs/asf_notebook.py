import os # for chdir, getcwd, path.exists 
import time # for perf_counter
import requests # for post
from getpass import getpass
import json # for json


# Takes a string path, returns true if exists or 
# prints error message and returns false if it doesn't.
def pathExists(path):
    if os.path.exists(path):
        return True
    else:
        print(f"Invalid Path: {path}")
        return False


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
            params = [('granule_list', granule_name), ('output', 'json')], 
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
    start = time.perf_counter()
    with open(local_filename, 'wb') as f:
        if r is None:
            f.write(r.content)
        else:
            dl = 0
            for chunk in r.iter_content(chunk_size=1024*1024):     
                dl += len(chunk)
                if chunk: # filter out keep-alive new chunks                   
                    f.write(chunk)
                    f.flush()
                    done = int(50 * dl / int(total_length))
                    print("\r[%s%s] %s bps, %s%%    " % ('=' * done, ' ' * (50-done), dl//(time.perf_counter() - start), int((100*dl)/total_length)), end='\r', flush=True)    
        if os.stat(local_filename).st_size < total_length:
            print('\nDownload failed!\n')
            return
        else:
            print('\nDone\n')
            return local_filename
