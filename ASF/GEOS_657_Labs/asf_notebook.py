import os # for chdir, getcwd, path.exists 
import time # for perf_counter
import requests # for post
from getpass import getpass
import json # for json
import zipfile # for extractall, BadZipFile


# Takes a string path, returns true if exists or 
# prints error message and returns false if it doesn't.
def path_exists(path):
    if os.path.exists(path):
        return True
    else:
        print(f"Invalid Path: {path}")
        return False

    
def ASF_unzip(directory_path, file_path ):
    file_name, ext = os.path.splitext(file_path)
    if ext == ".zip":
        print(f"Extracting: {file_path}")
        try:
            zipfile.ZipFile(file_path).extractall(directory_path)
        except zipfile.Zipfile.BadZipFile:
            print(f"Zipfile Error.")
        return
    

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
        

#######################
#  Hyp3 API Functions #
#######################

# get_hyp3_subscriptions
# Takes a Hyp3 API object and returns a list of enabled associated subscriptions
# Returns None if there are no enabled subscriptions associated with Hyp3 account.
# precondition: must already be logged into hyp3
def get_hyp3_subscriptions(hyp3_api_object):
    subscriptions = hyp3_api.get_subscriptions(enabled=True)
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
                    print("Invalid ID\nPick a subscription ID from the above list.")
                    clear_output()
            if subscription_id in subscription_ids: 
                break
            else:
                print("Invalid ID\nPick a subscription ID from the above list.")
                clear_output()
        return subscription_id
                          
                   
# download_hyp3_products()
# Takes a Hyp3 API object and a destination path.
# preconditions: 
# -must already be logged into hyp3
# -path must be valid                          
def download_hyp3_products(hyp3_api_object, path):
    subscriptions = get_hyp3_subscriptions(hyp3_api_object)
    subscription_id = pick_hyp3_subscription(subscriptions)
    if subscription_id:
        products = hyp3_api.get_products(sub_id=subscription_id)
        if path_exists(products_path):
            for p in products:
                url = p['url']
                _match = re.match(r'https://hyp3-download.asf.alaska.edu/asf/data/(.*).zip', url)
                product = _match.group(1)
                filename = f"{path}/{product}"
                if not os.path.exists(filename): # if not already present, we need to download and unzip products
                    print(f"\n{product} is not present.\nDownloading from {url}")
                    r = requests.get(url, stream=True)
                    total_length = int(r.headers.get('content-length'))
                    with open(filename, 'wb') as f:
                        start = time.perf_counter()
                        dl = 0
                        for chunk in r.iter_content(chunk_size=1024*1024):
                            dl += len(chunk)
                            if chunk:
                                f.write(chunk)
                                f.flush()
                                done = int(50 * dl / int(total_length))
                                print("\r[%s%s] %s bps, %s%%    " % ('=' * done, ' ' * (50-done), dl//(time.perf_counter() - start), int((100*dl)/total_length)), end='\r', flush=True)    
            print(f"\n")
            os.rename(filename, f"{filename}.zip")
            filename = f"{filename}.zip"
            ASF_unzip(path, filename)
            os.remove(filename)
            print(f"\nDone.")                       
                        
