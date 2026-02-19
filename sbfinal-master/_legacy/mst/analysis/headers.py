#create headers for output files
import json


def make_header(userfile): # summarize key userfile values

    input = userfile.INPUT  # INPUT section of userfile
    str0 = str(input)
    print("str0: ", str0)

    hdrtxt = json.dumps(str0)
    print("hdrtxt: ", hdrtxt)

    print("Items in Inputs: ")
    kys = input.keys()
    for key in kys:
        #print("item: ", item)
        val = input.get(key)
        print( key, ': ',val)


    settings = userfile.SETTINGS   # SETTINGS section of userfile
    str1 = str(settings)
    print("str1: ", str1)

    hdrtxt = json.dumps(str1)
    print("hdrtxt: ", hdrtxt)

    print("Items in Settings: ")
    kys = settings.keys()
    for key in kys:
        #print("item: ", item)
        val = settings.get(key)
        print( key, ': ',val)


    state = userfile.STATE  # STATE section of userfile
    str2 = str(state)
    print("str2: ", str2)

    hdrtxt = json.dumps(str2)
    print("hdrtxt: ", hdrtxt)

    print("Items in State: ")
    kys = state.keys()
    for key in kys:
        #print("item: ", item)
        val = state.get(key)
        print( key, ': ',val)


    hdr = str0 + '\n' + str1 + '\n' + str2

    return hdr
