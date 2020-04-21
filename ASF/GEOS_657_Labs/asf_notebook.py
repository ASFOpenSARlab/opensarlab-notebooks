# asf_notebook.py
# Alex Lewandowski
# 3-30-20
# Module of Alaska Satellite Facility OpenSARLab Jupyter Notebook helper functions 


import math
import os  # for chdir, getcwd, path.exists
import re
import time  # for perf_counter
import requests  # for post, get
from getpass import getpass  # used to input URS creds and add to .netrc
import zipfile  # for extractall, ZipFile, BadZipFile
from datetime import datetime, date
import glob
import sys
import urllib
from subprocess import call, PIPE
import json

import gdal  # for Open
import numpy as np
import pandas as pd

from IPython.utils.text import SList
from IPython.display import clear_output
import ipywidgets as widgets
from ipywidgets import Layout

from asf_hyp3 import API, LoginError  # for get_products, get_subscriptions, login

from bokeh.plotting import figure
from bokeh.tile_providers import get_provider, Vendors
from bokeh.models import ColumnDataSource, GMapOptions, BoxSelectTool, HoverTool, CustomJSHover, CustomJS, Rect, Div, ResetTool, MultiPolygons
from bokeh.client import push_session
from bokeh.io import curdoc, output_notebook, push_notebook, show
from bokeh import events
from bokeh.models.glyphs import Rect


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

        
def get_power_set(my_set,set_size): 
    p_set = set()
    # set_size of power set of a set 
    # with set_size n is (2**n -1) 
    pow_set_size = (int) (math.pow(2, set_size)); 
    counter = 0; 
    j = 0; 
    # Run from counter 000..0 to 111..1 
    for counter in range(0, pow_set_size):
        temp = ""
        for j in range(0, set_size): 
              
            # Check if jth bit in the  
            # counter is set If set then  
            # print jth element from set 
            if((counter & (1 << j)) > 0):
                if temp != "":
                    temp = f"{temp} and {my_set[j]}"
                else:
                    temp = my_set[j]
            if temp != "":
                p_set.add(temp)
    return p_set        
        
    
def remove_nan_filled_tifs(tif_dir: str, file_names: list):
    """
    Takes a path to a directory containing tifs and
    and a list of the tif filenames.
    Deletes any tifs containing only NaN values.  
    """
    assert type(tif_dir) == str, 'Error: tif_dir must be a string'
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
    
    
def input_path(prompt):        
    print(f"Current working directory: {os.getcwd()}") 
    print(prompt)
    return input()


def handle_old_data(data_dir, contents):
    print(f"\n********************** WARNING! **********************")
    print(f"The directory {data_dir} already exists and contains:")
    for item in contents:
        print(f"• {item.split('/')[-1]}")
    print(f"\n\n[1] Delete old data and continue.")
    print(f"[2] Save old data and add the data from this analysis to it.")
    print(f"[3] Save old data and pick a different subdirectory name.")
    while True:
        try:
            selection = int(input("Select option 1, 2, or 3.\n"))
        except ValueError:
             continue
        if selection < 1 or selection > 3:
             continue
        return selection

              
###################
#  GDAL Functions #
###################
              
def vrt_to_gtiff(vrt: str, output: str):
    if '.vrt' not in vrt:
        print('Error: The path to your vrt does not contain a ".vrt" extension.')
        return
    if '.' not in output:
        output = f"{output}.tif"
    elif len(output) > 4 and (output[:-3] == 'tif' or output[:-4] == 'tiff'):
        print('Error: the output argument must either not contain a ' /
              'file extension, or have a "tif" or "tiff" file extension.')
        return
        
    cmd = f"gdal_translate -co \"COMPRESS=DEFLATE\" -a_nodata 0 {vrt} {output}"
    sub = subprocess.run(cmd, stderr=subprocess.PIPE, shell=True)
    print(str(sub.stderr)[2: -3])
              
########################
#  Earth Data Function #
########################

class EarthdataLogin:
  
    def __init__(self):
              
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
                    self.username = username
                    self.password = password
                    self.api = api
                    break


    def login(self):
        try: 
            self.api.login(self.password)
        except LoginError:
            raise
                   

#########################
#  Vertex API Functions #
#########################
              
def get_vertex_granule_info(granule_name: str, file_type=None) -> dict:
    """
    Takes a string granule name and string file type (deprecated), and returns the granule info as json.
    preconditions:
    Requires AWS Vertex API authentification (already logged in).
    Requires a valid granule name.
    """
    assert type(granule_name) == str, 'Error: granule_name must be a string.'

    vertex_API_URL = "https://api.daac.asf.alaska.edu/services/search/param"
    try:
        if file_type:
            print("Deprecation Warning: The file_type parameter has been deprecated and will "
                  "be removed in a future release.")
        response = requests.post(
                vertex_API_URL,
                params=[('granule_list', granule_name), ('output', 'json')]
            )
    except requests.exceptions.RequestException as e:  # This is the correct syntax
        print(e)
        sys.exit(1)
    else:
        if len(response.json()) > 0:
            try:
                json_response = response.json()[0][0]
                return json_response
            except IndexError as e:
              print(f"Error: {granule_name} does not appear to be a valid scene name.")
              return
        else:
            print("get_vertex_granule_info() failed.\ngranule/processing level mismatch.")
              

#######################
#  Hyp3 API Functions #
#######################

                        
def get_hyp3_subscriptions(login: EarthdataLogin) -> dict:
    """
    Takes an EarthdataLogin object and returns a list of associated, enabled subscriptions
    Returns None if there are no enabled subscriptions associated with Hyp3 account.
    """
    
    assert type(login) == EarthdataLogin, 'Error: login must be an EarthdataLogin object'    
    
    while True:
        subscriptions = login.api.get_subscriptions(enabled=True)
        try:
            if subscriptions['status'] == 'ERROR' and \
                  subscriptions['message'] == 'You must have a valid API key':
                creds = login.api.reset_api_key()
                login.api.api = creds['api_key']
        except (KeyError, TypeError):
            break

    if not subscriptions:
        print("There are no subscriptions associated with this Hyp3 account.")
    else:
        subs = []
        for sub in subscriptions:
            subs.append(f"{sub['id']}: {sub['name']}")
    return subs                        
                        
            
def get_subscription_products_info(subscription_id: int, login: EarthdataLogin) -> list:
                        
    assert type(subscription_id) == str, f'Error: subscription_id must be a string, not a {type(subscription_id)}'                      
    assert type(login) == EarthdataLogin, f'Error: login must be an EarthdataLogin object, not a {type(login)}'                     
                        
    products = []
    page_count = 0
    while True:       
        product_page = login.api.get_products(
            sub_id=subscription_id, page=page_count, page_size=100)
        try:
            if product_page['status'] == 'ERROR'and \
                  product_page['message'] == 'You must have a valid API key':
                creds = login.api.reset_api_key()
                login.api.api = creds['api_key']
                continue
        except (KeyError, TypeError):
            page_count += 1           
            pass
        if not product_page:
            break
        for product in product_page:
            products.append(product)
    return products        

            
def get_product_info(products_info: list, date_range: list) -> dict:               
    paths = []
    directions = []
    urls = []
    vertex_API_URL = "https://api.daac.asf.alaska.edu/services/search/param"
    for i, p_info in enumerate(products_info):
        if p_info['process_id'] == 32 and i == len(products_info) - 1:
            break
        dt = p_info['name'].split('_')[4].split('T')[0]
        if date(int(dt[:4]), int(dt[4:6]), int(dt[-2:])) >= date_range[0]:
            if date(int(dt[:4]), int(dt[4:6]), int(dt[-2:])) <= date_range[1]:
                granule_name = p_info['name'].split('-')[0]
                parameters = [('granule_list', granule_name), ('output', 'json')]
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
                paths.append(json_response['track'])
                directions.append(json_response['flightDirection'])
                urls.append(p_info['url'])
    return {'paths': paths, 'directions': directions, 'urls': urls}           

            
def get_products_dates(products_info: list) -> list:
    dates = []
    for info in products_info:
        for chunk in info['name'].split('_'):
            if len(chunk) == 15 and 'T' in chunk:
                dates.append(chunk[:8])
                break
    dates.sort()
    return dates
            
            
def get_products_dates_insar(products_info: list) -> list:
    dates = []
    for info in products_info:
        dates.append(info['name'].split('-')[1].split('T')[0])
        dates.append(info['name'].split('-')[2].split('T')[0])
    dates.sort()
    return dates      
         
            
def gui_date_picker(dates: list) -> widgets.SelectionRangeSlider:  
    start_date = datetime.strptime(min(dates), '%Y%m%d')
    end_date = datetime.strptime(max(dates), '%Y%m%d')
    date_range = pd.date_range(start_date, end_date, freq='D')
    options = [(date.strftime(' %m/%d/%Y '), date) for date in date_range]
    index = (0, len(options)-1)
    
    selection_range_slider = widgets.SelectionRangeSlider(
    options = options,
    index = index,
    description = 'Dates',
    orientation = 'horizontal',
    layout = {'width': '500px'})
    return(selection_range_slider)  
          
                     
def get_slider_vals(selection_range_slider: widgets.SelectionRangeSlider) -> list:
    '''Returns the minimum and maximum dates retrieved from the
    interactive time slider.
    
    Parameters:
    - selection_range_slider: Handle of the interactive time slider
    '''
    [a,b] = list(selection_range_slider.value)
    slider_min = a.to_pydatetime()
    slider_max = b.to_pydatetime()
    return[slider_min, slider_max]        
                                           
            
def get_RTC_polarizations(base_path: str) -> list:
    """
    Takes a string path to a directory containing RTC product directories
    Returns a list of present polarizations
    """
    assert type(base_path) == str, 'Error: base_path must be a string.'
    assert os.path.exists(base_path), f"Error: select_RTC_polarization was passed an invalid base_path, {base_path}"
    paths = []
    pths = glob.glob(f"{base_path}/*/*.tif")
    if len(pths) > 0:
        for p in pths:
            filename = os.path.basename(p)
            polar_fname = re.search("^\w{20,80}(_|-)(vv|VV|vh|VH|hh|HH|hv|HV).tif$", filename)
            if polar_fname:
                paths.append(polar_fname.string.split('.')[0][-2:])
    if len(paths) > 0:
        return list(set(paths))
    else:
        print(f"Error: found no available polarizations.")                          
         
   
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

            
            
def select_parameter(name: str, things: set):
    return widgets.RadioButtons(
        options=things,
        description=name,
        disabled=False,
        layout=Layout(min_width='800px')
    )

 
            
def select_mult_parameters(name: str, things: set):
    height = len(things) * 19
    return widgets.SelectMultiple(
        options=things,
        description=name,
        disabled=False,
        layout=widgets.Layout(height=f"{height}px", width='175px')
    )                      
            
def get_wget_cmd(url: str, login: EarthdataLogin) -> str:
    cmd = f"wget -c -q --show-progress --http-user={login.username} --http-password={login.password} {url}"
    return cmd          
            
        
########################################
#  Bokeh related Functions and Classes #
########################################
            
def remote_jupyter_proxy_url(port):
    """
    Callable to configure Bokeh's show method when a proxy must be
    configured.

    If port is None we're asking about the URL
    for the origin header.
    """   
    #base_url = os.environ['EXTERNAL_URL']
    base_url = 'https://opensarlab.asf.alaska.edu/'
    host = urllib.parse.urlparse(base_url).netloc

    # If port is None we're asking for the URL origin
    # so return the public hostname.
    if port is None:
        return host

    service_url_path = os.environ['JUPYTERHUB_SERVICE_PREFIX']
    proxy_url_path = 'proxy/%d' % port                  

    user_url = urllib.parse.urljoin(base_url, service_url_path)
    full_url = urllib.parse.urljoin(user_url, proxy_url_path)
    return full_url
            
       
class AOI:
    def __init__(self, 
                 lower_left_coord=[-20037508.342789244, -19971868.880408563], 
                 upper_right_coord=[20037508.342789244, 19971868.880408563]):
        
        e_list = "Passed coordinates must be a list"
        assert type(lower_left_coord) == list, e_list   
        assert type(upper_right_coord) == list, e_list
        
        e_length = "Error: lower_left_coord must contain one EPSG:3857 coordinate [x, y]"
        assert len(lower_left_coord) == 2, e_length
        assert len(upper_right_coord) == 2, e_length
        
        e_order = "Error: A lower_left_coord value is greater than an upper_right_coord value."
        assert lower_left_coord[0] < upper_right_coord[0], e_order
        assert lower_left_coord[1] < upper_right_coord[1], e_order
        
        coord_error = False
        e_off_planet = "Error: Cannot instantiate AOI class object with invalid EPSG:3857 coordinates."
        if lower_left_coord[0] < -20037508.342789244 or lower_left_coord[0] > 20037508.342789244:
            coord_error = True
        if upper_right_coord[0] < -20037508.342789244 or upper_right_coord[0] > 20037508.342789244:
            coord_error = True
        if lower_left_coord[1] < -19971868.880408563 or lower_left_coord[1] > 19971868.880408563:
            coord_error = True
        if upper_right_coord[1] < -19971868.880408563 or upper_right_coord[1] > 19971868.880408563:
            coord_error = True 
        if coord_error:
            assert False, e_off_planet
        
        self.geom = {}
        self.tiff_stack_coords = [lower_left_coord, upper_right_coord]
        self.subset_coords = [[None, None], [None, None]]
        self.p = None
        self.sources = {}
        self.callbacks = {}
        
        self.create_sources()
        self.create_callbacks()
        
        
        
    def update_subset_bounds(self, attributes=[]):
        def python_callback(event):
            self.geom.update(event.__dict__['geometry'])
            #print(event.__dict__['geometry'])
            self.subset_coords[0][0] = event.__dict__['geometry']['x0']
            self.subset_coords[0][1] = event.__dict__['geometry']['y0']
            self.subset_coords[1][0] = event.__dict__['geometry']['x1']
            self.subset_coords[1][1] = event.__dict__['geometry']['y1']
            print("\rAOI.subset_coords: [[%s, %s], [%s, %s]]      " % (self.subset_coords[0][0], 
                                                                       self.subset_coords[0][1], 
                                                                       self.subset_coords[1][0], 
                                                                       self.subset_coords[1][1]), 
                                                                      end='\r', flush=True
                 )
            
        return python_callback
    
    
    def reset_subset_bounds(self):
        self.subset_coords = [[None, None], [None, None]]
      
    
    def create_callbacks(self):
        subset = CustomJS(args=dict(source=self.sources['subset']), code="""
            // get data source from Callback args
            var data = source.data;

            /// get BoxSelectTool dimensions from cb_data parameter of Callback
            var geometry = cb_data['geometry'];

            var x0 = geometry['x0'];
            var y0 = geometry['y0'];
            var x1 = geometry['x1'];
            var y1 = geometry['y1'];
            var xxs = [[[x0, x0, x1, x1]]];
            var yys = [[[y0, y1, y1, y0]]];

            /// update data source with new Rect attributes
            data['xs'].pop();
            data['ys'].pop();
            data['xs'].push(xxs);
            data['ys'].push(yys);

            // emit update of data source
            source.change.emit();
        """)
        
        latitude = CustomJSHover(code="""
                        var projections = require("core/util/projections");
                        var x = special_vars.x
                        var y = special_vars.y
                        var coords = projections.wgs84_mercator.inverse([x, y])
                        return "" + coords[1].toFixed(6)
                    """)
        
        longitude = CustomJSHover(code="""
                        var projections = require("core/util/projections");
                        var x = special_vars.x
                        var y = special_vars.y
                        var coords = projections.wgs84_mercator.inverse([x, y])
                        return "" + coords[0].toFixed(6)
                    """)    

        self.callbacks.update([('subset', subset), 
                               ('latitude', latitude), 
                               ('longitude', longitude)])


    def create_sources(self):
        empty = np.array([np.linspace(0, 0, 2)]*2) #the empty image data to which the HoverTool is attached

        lx = -20037508.342789244 #min web mercator lat
        ly = -19971868.880408563 #min web mercator long
        # stretch empty image across world map so lat, long hover still works if user zooms out of AOI
        hover_img = dict(image=[empty],
                    x=[lx],
                    y=[ly],
                    dw=[int(lx*-2)],
                    dh=[int(ly*-2)])

        subset = ColumnDataSource(data=dict(xs=[], ys=[]))
            
        self.sources.update([('hover', hover_img), 
                               ('subset', subset)])
    
    
    def build_plot(self, doc):
        tile_provider = get_provider('STAMEN_TERRAIN')
        box_select = BoxSelectTool(callback=self.callbacks['subset'])
            
        self.p = figure(title="Use The Square Selection Tool To Select An Area Of Interest",
                   x_range=(self.tiff_stack_coords[0][0]-10000, self.tiff_stack_coords[1][0]+10000), 
                   y_range=(self.tiff_stack_coords[0][1]-10000, self.tiff_stack_coords[1][1]+10000),
                   x_axis_type="mercator", 
                   y_axis_type="mercator",
                   tools=['reset', box_select, 'pan', 'wheel_zoom', 'crosshair'])

        
        hover_img = self.p.image(source=self.sources['hover'], 
                                 image='image', 
                                 x='x', y='y', 
                                 dw='dw', dh='dh', 
                                 alpha=0.0)

        self.p.add_tools(HoverTool(
            renderers=[hover_img],
            tooltips=[
                ( 'Long',  '@x{custom}'),
                ( 'Lat',   '@y{custom}'  )],
            formatters=dict(
                y=self.callbacks['latitude'],
                x=self.callbacks['longitude'])
        ))

        self.p.add_tile(tile_provider)

        x1 = self.tiff_stack_coords[0][0]
        x2 = self.tiff_stack_coords[1][0]
        y1 = self.tiff_stack_coords[0][1]
        y2 = self.tiff_stack_coords[1][1]
        self.p.multi_polygons(xs=[[[[x1, x1, x2, x2]]]],
                             ys=[[[[y1, y2, y2, y1]]]],
                             line_width=1.5, line_color='black',
                             fill_color=None)

        self.p.js_on_event(events.Reset, self.reset_subset_bounds())
        self.p.on_event(events.SelectionGeometry, 
                        self.update_subset_bounds(attributes=['geometry'])
                       )
        
        subset = MultiPolygons(xs='xs', ys='ys',
                               fill_alpha=0.15, fill_color='#336699',
                               line_dash='dashed'
                              )

        glyph = self.p.add_glyph(self.sources['subset'], 
                                 subset, 
                                 selection_glyph=subset, 
                                 nonselection_glyph=subset)
        
        doc.add_root(self.p)
        
    def display_AOI(self):        
        #output_notebook()
        show(self.build_plot, notebook_url=remote_jupyter_proxy_url)
        print("Selected bounding box coords stored in AOI.subset_coords")
        print("[[lower_left_x, lower_left_y], [upper_right_x, upper_right_y]]\n")
  
