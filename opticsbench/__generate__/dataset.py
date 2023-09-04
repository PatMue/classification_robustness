
import os 

import shutil

from tqdm import tqdm 


root = r"/corruptions"

copy = False 
remove_folder = False


for corruption in os.listdir(root):
	
	fpath = os.path.join(root,corruption)
	
	for sev in tqdm(os.listdir(fpath)):
		
		f = os.path.join(fpath,sev)
		
		folders = os.listdir(f)


		# now further limit file amount to another 1/20 ...
		
		selection = folders[::20]

		for folder in folders:
			if not folder in selection:
				os.remove(os.path.join(f,folder))


		# now remove if folder:			
		if remove_folder:
			for folder in folders:
				ff = os.path.join(f,folder)
				if os.path.isdir(ff):
					#input(ff)
					shutil.rmtree(ff)
		
		
		if copy:					
			for folder in folders:			
				files = os.listdir(ff)[::10] # every 20th image 
				
				for file in files:
					p = os.path.join(ff,file)
					shutil.copyfile(p,os.path.join(f,file))
			
			