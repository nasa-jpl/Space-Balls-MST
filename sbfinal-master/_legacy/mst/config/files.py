import os, sys


sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(__file__))


# specify filenames here

# atmosphere model uses this file
#AtmModel = os.path.dirname(__file__) + '/CurF10_Apr2021.txt'
#AtmModel = os.path.dirname(__file__) + '/CurF10_Mar2023.txt'
AtmModel = os.path.dirname(__file__) + '/CurF10_extended.txt'
# '../../../Users/reynerso/Documents/Monte/CurF10_Apr2021.txt'
#AtmModel = '/home/reynerso/Python/CurF10_extended.txt'
#AtmModel = '/home/reynerso/SphereX/CurF10_Apr2021.txt'

#filename for a boa output
outputName = 'OCO2_out.boa'

#filename for an STK format output
stkoutfileName = 'stkout_17Jun24.e'

# these were run for setup - on nexus
#execfile('/home/reynerso/Python/setupMonteMD.py')
#execfile('/home/reynerso/Python/setupOsmean.py')

# loading boa files...
#boa.load('../../../Users/reynerso/Documents/Monte/TrajectoryFiles/de421.boa') # see loads of boas in boas.py
#boa.load('/nav/common/import/ephem/de421.boa')


#def file_path( path ):  # file path designation

def last_Xchars(x):  # set to sort for last 6 chars
    return ( x[-16:] )

def file_sort(folder):  # sorts files by last X chars in a given directory;  returns a list that has sorted order
    file_list = os.listdir( folder)
    sorted( file_list, key = last_Xchars)
    return file_list 

def getint(name):
    print("name: ",name)
    #basename = name.partition('.')

    basename = name.split('_')
    #(alpha1,alpha2,num0, num, alpha3) = basename.split('_')
    num = basename[3]
    return int(num)

def checkcsv(name):
    basename = name.split('.')
    if len(basename) == 2:
        if basename[1] == 'csv':
            print('found csv: ', name)
            return 1 
        else: return 0       
    else: return 0

def file_sort2(dir_name):  # sorts files by last X chars in a given directory;  returns a list that has sorted order
    #dir_name = os.listdir( folder)

    list_of_files = sorted( filter( lambda x: os.path.isfile(os.path.join(dir_name, x)), os.listdir(dir_name) ) )

    # Iterate over sorted list of files and print the file paths 
    # one by one.
    for file_path in list_of_files:
        print(file_path)


    newfilelist = []
    for file in list_of_files:
        check = checkcsv(file)
        if check:
            newfilelist.append(file)

    print("")    
    print("newfilelist:")
    print(newfilelist)

    return newfilelist 


if __name__ == "__main__":

    '''

    #folder = '~/Documents/Python/Spaceball/data/hourly/sw'

    folder = '/home/monte/Documents/Python/Spaceball/data/hourly/sw'
    #folder = os.path.dirname(__file__) 
    #sys.path.insert(0, os.path.dirname( folder ))
    file_list = os.listdir( folder)
    print("UNsorted listing:")
    print( file_list )    
    print("")

    filelist = file_sort( folder )
    print("folder: ", folder)
    for file in filelist:
        print("file: ",file)
    print("")
    print("sorted listing:")
    print(sorted( filelist, key = last_Xchars))

    
    import glob
    #import os
    '''

    #dir_name = '/Users/reynerso/Documents/Python/Spaceball/data/hourly/sw'
    dir_name = '/home/monte/Documents/Python/Spaceball/data/hourly/sw'    


    print( "dir name: ", dir_name )

    # Get list of all files in a given directory sorted by name
    #list_of_files = sorted( filter( os.path.isfile,  glob.glob(dir_name + '*') ) )

    list_of_files = sorted( filter( lambda x: os.path.isfile(os.path.join(dir_name, x)), os.listdir(dir_name) ) )

    # Iterate over sorted list of files and print the file paths 
    # one by one.
    for file_path in list_of_files:
        print(file_path)

    ''' 
    count = 0 

    print( "using Count method to index the list...")

    while count < 25:
        print( list_of_files[count] ) 
        count = count + 1
    
    print( " length of list: ", len(list_of_files)  )

    #list_of_files.sort(key=lambda f: int(filter(str.isdigit, f)))
    #list_of_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    #sort(key=lambda f: int(re.sub('\D', '', f)))
    '''

    newfilelist = []
    for file in list_of_files:
        check = checkcsv(file)
        if check:
            newfilelist.append(file)

    print("")    
    print("newfilelist:")
    print(newfilelist)
    '''
    list_of_files.sort(key=getint)
    print("")
    print("Another file sort method...")
    print(list_of_files)

    print( " length of list: ", len(list_of_files)  )
    '''
    print("")
    print("execute file_sort2 on :", dir_name)

    nfl = file_sort2(dir_name)

    dir_name2 = '/home/monte/Documents/Python/Spaceball/data/hourly/lw'  

    print("")
    print("execute file_sort2 on: " , dir_name2)

    nfl2 = file_sort2(dir_name2)








