# Understanding pandas files system and making a dataframe
#Unpacking and contacting multiple files in the panda frame
#Lets suppose you have multiple files and you want to convert them 
#into the panda frame. This will import all the CSV files into the pandas frame, 
#now lets read the code, before that since importing multiple files, 
#is a recursive function, so lets understand the glob function in python, 
#which searches the files recursively

glob function --->  glob.glob(pathname, *, recursive=False), 

#by default the recursive is set to false, means it will not search the files 
#recursively, by you can set the recursive to TRUE then it will 
#match True “**” followed by path separator('./**/').
# (geeksforgeerks.org). 

#This is the same as the find function in bash: find . -iname filename 
#(listing all the files in the directory . with the name given in the -iname).

#Returns a list of names in list files.
# check out the ** which is indicating the recursive true.
import glob
print("Using glob.glob()")
files = glob.glob('/Desktop/**/*.txt', recursive = True)  
for file in files:
    print(file)

#It returns an iterator which will be printed simultaneously.
print("\nUsing glob.iglob()")
for filename in glob.iglob('/Desktop/**/*.txt',recursive = True):
    print(filename)

#Another way to walk through a directory and listing all the files in the
import os
for dirname, _, filenames in os.walk('../Desktop/data/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#Now if you want to search with the file name or the wildcard like you do in bash 
#To list all the files or the
     $ find . -iname *
#Similarly you can list all the files in the directly
$ find . -iname filename 
# calling the files with the name tag 
import glob
for name in glob.glob('/home/data/data.txt'): 
    print(name)
 	Using '*' pattern 
# calling all the files in the directory
for name in glob.glob('/home/data/*'): 
    print(name)
  	Using '?' pattern
# calling all the files with the wildcard 
for name in glob.glob('/home/data/data?.txt'): 
    print(name)
	Using [0-9] pattern
# calling all the files with the number tag.
for name in glob.glob('/home/data/*[0-9].*'):  
    print(name)  
#Now after finding all the files list, its time to make a dataframe out of it. 
#Making a data frame from multiple CSV files with in a folder
import glob
import os
import pandas as pd
for name in glob.glob('/home/data/*'): # calling all the files in the directory
    print(name)
all_files = glob.glob("folder/*.csv") 
df = pd.concat((pd.read_csv(f) for f in all_files))
print(df)
#Making a data from from multiple CSV files from a specified folder, 
#where PATH_TO_FOLDER indicates the PATH. 
home = os.path.expanduser("~")
path = f"{home}PATH_TO_FOLDER"
for name in glob.glob('/home/data/*'): # calling all the files in the directory
    print(name)
all_files = glob.glob(path + "/*.csv")
df = pd.concat((pd.read_csv(f) for f in all_files))
print(df)
#Making a data frame from the specificed folder which can be a nested folder. 
path = f"{home}PATH_TO_FOLDER"
for name in glob.glob('/home/data/*'): # calling all the files in the directory
    print(name)
all_files = glob.glob(path + "/**/*.csv")
df = pd.concat((pd.read_csv(f) for f in all_files))
print(df)
#To get the path of the foler you can use os.getcwd() 
#and to change the directory you can use os.chdir().
import os
print(os.getcwd())
os.chdir('/Users/name/Documents')
print(os.getcwd())
#Taking into above steps: a complete functional code for the 
#merging panda frame from multiple CVS will be 
import os
print(os.getcwd())
os.chdir('/Users/name/Documents')
print(os.getcwd())
import glob
import os
import pandas as pd
for name in glob.glob('/home/data/*'): # calling all the files in the directory
    print(name)
all_files = glob.glob("folder/*.csv") 
df = pd.concat((pd.read_csv(f) for f in all_files))
print(df)
