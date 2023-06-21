#!/usr/bin/env python
# coding: utf-8
'''UAV SAR Download API
Software for Earth Big Data Processing, Prediction Modeling and Organzation (SEPPO) 
(c) 2020 Earth Big Data LLC
Author: Josef Kellndorfer, 
Date: 2020-01-30
'''
from __future__ import print_function
try:
	from  urllib.request import  urlopen  # Python 3
	# print('Python 3')
except:
	from  urllib2 import urlopen  # Python >= 2.7
	# print('Python 2')
import os,sys

dl_tries=5  # Hardcode on how many times curl tries to download


def myargsgarse(a):
	import argparse
	class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
			pass
	thisprog=os.path.basename(a[0])

	epilog='''
	\r************
	\r* Examples *
	\r************

	\r**** LIST all data takes 
	\r{thisprog} -ldt 
	\r{thisprog} -ldt --campaign NISARA
	\r{thisprog} -ldt --campaign NISARA --site 06800

	\r**** Download all datatakes for a campaign and site
	\r{thisprog} --campaign NISARA --site 06800 --DryRun

	\r**** Download specific datatakes for a campaign and site
	\r{thisprog} --campaign NISARA --site 06800 --datatakes 19043_002 19070_004 --DryRun

	\r**** Download all datatakes and dems for a campaign and site 
	\r{thisprog} --campaign NISARA --site 06800 --dem 0 --DryRun

	\r**** Download all datatakes and dem from third data take for a campaign and site 
	\r{thisprog} --campaign NISARA --site 06800 --dem 3 --DryRun

	\r**** Download all datatakes and dem from third data take for a campaign and site and selected polarizations
	\r{thisprog} -campaign NISARA --site 13905 -dem 1 -p HHHH VVVV HVHV --DryRun

	'''.format(thisprog=thisprog)

	p = argparse.ArgumentParser(prog=thisprog,description="Earth Big Data's download API for UAVSAR data",epilog=epilog,formatter_class=CustomFormatter)
	p.add_argument("-u","--urlroot",required=False, help="ASF DAAC or JPL URLROOT", action='store',default='https://uavsar.asf.alaska.edu')
	# p.add_argument("-u","--urlroot",required=False, help="ASF DAAC or JPL URLROOT", action='store',default='https://uavsar.jpl.nasa.gov')
	p.add_argument("-i","--index_file",required=False, help="Locally stored index_file", action='store',default=os.path.join('.uavsar','index.html'))
	p.add_argument("-d","--datatakes",required=False, nargs='*', help="List of datatakes. If none, get all. Can be any substring after the 'campaign_site' portion of the data take name", action='store',default=None)
	p.add_argument("-s","--site",required=False, help="UAVSAR Site code e.g. 06800'",action='store',default='')
	p.add_argument("-c","--campaign",required=False, help="NISAR Campaign, .e.g. 'NISARA'",action='store',default='')
	p.add_argument("-p","--polarizations", nargs='*',required=False, help="Polariztions",action='store',default=['HHHH','HVHV'])
	p.add_argument("-t","--image_types",required=False, nargs='+',help="Image Types",action='store',default=['mlc'],choices=['mlc','slc','grd'])
	p.add_argument("-dem","--dem",required=False, help="Download DEM Number from site, number indicates which time step to get, 0 download all. Defaults to no dem download",action='store',default=None,type=int)
	p.add_argument("-na","--no_annotation",required=False, help="Do not download the annotationfile .ann",action='store_true',default=False)
	p.add_argument("-l","--list_only",required=False, help="only list the urls, do not download",action='store_true',default=False)
	p.add_argument("-ldt","--list_datatakes",required=False, help="List all data takes. Can be combined with --campaign --site",action='store_true',default=False)
	p.add_argument("-r","--rebuild_index",required=False, help="Rebuilds the local index file from -u",action='store_true',default=False)
	p.add_argument("-dryrun","--DryRun",required=False, help="DryRun",action='store_true',default=False)
	p.add_argument("-v","--verbose",required=False, help="Verbose output",action='store_true',default=False)
	p.add_argument("outdir_root",nargs='?',help="Output directory root. Sub-directories with campain_site will be generated",default=os.path.join('Downloads','UAVSAR'))
	args=p.parse_args(a[1:])

	if not args.site and not args.campaign and not args.list_datatakes:
			print('Need --site and --campaign  or --list_datatakes.\nFor help:\n{} -h'.format(thisprog))
			sys.exit(1)

	return args

def rebuild_index(args):
	'''Connects to ASF DAAC or JPL to retrieve the top level index.html file 
	which will be parsed to retrieve a list of all campaings and sites'''
	urlroot=args.urlroot
	with urlopen(urlroot) as res:
		res2=res.read().decode()
	if not os.path.exists(os.path.dirname(args.index_file)):
		os.makedirs(os.path.dirname(args.index_file))
	with open(args.index_file,"w") as f:
		f.write(res2)

def get_datatakes_from_index(idx_file):
	'''reads the local index.html file 
	which will be parsed to retrieve a lists of all campaings and sites'''
	with open(idx_file,"r") as f:
		lines=f.readlines()
	uavsar_takes=[x.split('href="')[1].split('/')[0] for x in lines if x.find('href="UA')>-1]
	return uavsar_takes

def get_download_urls(datatakes,args):
	urlroot      =args.urlroot
	datatake_list=args.datatakes
	campaign     =args.campaign
	site         =args.site
	polarizations=args.polarizations
	image_types  =args.image_types
	no_ann       =args.no_annotation
	dem_datatake =args.dem

	campaign_site=campaign+'_'+site
	dt = [x for x in datatakes if x.find(campaign_site)>-1]
	if datatake_list:
		dt = [x for x in dt if [y for y in datatake_list if x.find(y) > -1]]

	allfiles={}
	j=0
	try:
		for d in dt:
			res =  urlopen('/'.join([urlroot,d]))
			res2=res.read().decode()
			res.close()
			res=None
			dt_urls = ['/'.join([urlroot,d,x.split('">')[0]]) for x in res2.split('href="') if x.startswith(campaign_site)]
			if dt_urls:
				j+=1
				download_dem=False
				if dem_datatake!=None and (dem_datatake==0 or j==dem_datatake):
						download_dem=True
				dem_ann=[]
				if download_dem: dem_ann.append('hgt')
				if not no_ann: dem_ann.append('ann')
				if args.polarizations:
					final_files=[x for x in dt_urls if os.path.splitext(x)[1].replace('.','') in image_types]
					final_files=[x for x in final_files if [y for y in args.polarizations if x.find(y)>-1]]
				final_files+=[x for x in dt_urls if os.path.splitext(x)[1].replace('.','') in dem_ann]
				final_files.sort()
				allfiles[d]=final_files

		return allfiles
	except Exception as e:
		raise RuntimeError(e)

def download(dl_urls,outdir_root,campaign_site,DryRun=False,verbose=False):
	if DryRun:
		print("***** DRYRUN: No download. Would attempt to download files for {} datatakes".format(len(dl_urls)))
	try:
		outdir=os.path.join(outdir_root,campaign_site)
		if verbose or DryRun:
			print('Download Directory (created if not existing):',outdir)
		if not DryRun:
			if not os.path.exists(outdir):
				os.makedirs(outdir)
		# download command --wget works, curl does not.
		# dl_verbose  ='-v' if verbose else '-s'
		#getcmd = 'curl --fail --retry {} {} -C - -k -o'.format(dl_tries,dl_verbose)
		dl_verbose  ='-v' if verbose else '-q'
		getcmd = 'wget  -t {} {} -c --no-check-certificate -O'.format(dl_tries,dl_verbose)

		for d in dl_urls:
			outdir_d = os.path.join(outdir,d)
			if not DryRun:
				if not os.path.exists(outdir_d):
					os.makedirs(outdir_d)
			for url in dl_urls[d]:
				outfile=os.path.join(outdir_d,os.path.basename(url))
				cmd='{} {} {}'.format(getcmd,outfile,url).rstrip('\n')
				if verbose or DryRun:
					print('Downloading',url)
				if not DryRun:
					print(cmd)
					os.system(cmd)
	except Exception as e:
		raise RuntimeError(e)

def list_data_takes(datatakes,args):
	dt=datatakes
	if args.campaign:
		dt = [x for x in dt if x.split('_')[1]==args.campaign]
	if args.site:
		dt = [x for x in dt if x.split('_')[2]==args.site]
	for i in dt:
		if args.datatakes:
			if [x for x in args.datatakes if i.find(x) > -1]:
				print(i)
		else:
			print(i)



def processing(args):

	if args.rebuild_index or not os.path.exists(args.index_file):
		print('Building UAVSAR datatake index locally:',args.index_file)
		sys.stdout.flush()
		rebuild_index(args)

	datatakes=get_datatakes_from_index(args.index_file)

	if args.list_datatakes:
		list_data_takes(datatakes,args)
		sys.exit(1)

	dl_urls=get_download_urls(datatakes,args)

	if args.list_only:
		for i in dl_urls:
			print(i)
			for url in dl_urls[i]:
				print(url)
		sys.exit(1)

	campaign     =args.campaign
	site         =args.site
	campaign_site=campaign+'_'+site
	download(dl_urls,args.outdir_root,campaign_site,DryRun=args.DryRun,verbose=args.verbose)

def main(a):
	args=myargsgarse(a)
	processing(args)

if __name__ == '__main__':
	main(sys.argv)
