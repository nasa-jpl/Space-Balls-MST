'''
Settings for time, epochs, scenarios

'''

import Monte as M
import mpy.units as units
import math

class TimeHandler(object):

    # Initialize epoch and duration in Days (optional, default is 1 day)
    
    def __init__(self, start_epoch, duration=1.0 ):  #duration is in days
        super().__init__()
        self.start_epoch = start_epoch
        self.duration = duration
        self.stop_epoch = start_epoch + duration * 84600*units.sec
        self.interval = M.TimeInterval(start_epoch, self.stop_epoch) # create Monte Interval object for start and stop epochs

    def dur_fm_stop(self, stop_epoch): # make a duration object from self.start and an input stop epoch ; set the new Interval
        self.stop_epoch = stop_epoch
        self.duration = self.stop_epoch - self.start_epoch 
        self.interval = M.TimeInterval(self.start_epoch, self.stop_epoch) # create Monte Interval object for start and stop epochs

    def display(self):
        print("Timehandler object display start and stop epochs:")
        print("start_epoch: ",self.start_epoch)
        print("stop_epoch: ",self.stop_epoch)


def covert_Time_to_Epoch(time):  #Converts Time object in user_inputs.py to a Monte Epoch in UTC
    yr = str(time.year)
    mo = str(time.month)
    dy = str(time.day)
    hr = str(time.hour)
    mn = str(time.minute)
    sc = str(time.second)
    #Epoch format: M.Epoch('2021-Apr-1 00:00:00 UTC')
    epochstring = yr+'-'+mo+'-'+dy+' '+hr+':'+mn+':'+sc+' UTC'   
    epoch = M.Epoch(epochstring)
    print('epoch defined:',epochstring)
    print(epoch)
    return epoch

def convert_Time_to_DateStr(time): #Converts Time object in user_inputs.py to a Date string of form: 2018-01-01 or YYYY-MM-DD
    yr = str(time.year)
    mo = str(time.month)
    if time.month < 10:
        mo = "0"+mo
    dy = str(time.day)
    if time.day < 10:
        dy = "0"+dy
    datestring = yr+"-"+mo+"-"+dy
    return datestring

def convert_Epoch_to_DateStr(epoch):  #Converts Epoch object to a Date string of form: 2018-01-01 or YYYY-MM-DD
    print("Epoch to convert to datestring: ", epoch)
    cdateStartEpoch = M.CalDate(epoch) 
    year = str(cdateStartEpoch.year())
    month = str(cdateStartEpoch.month())
    day = str(cdateStartEpoch.day())
    #hr = str(cdateStartEpoch.hour())
    #mn = str(cdateStartEpoch.minute())
    #sec = str(cdateStartEpoch.second())
    #Epoch format: M.Epoch('2021-Apr-1 00:00:00 UTC')
    #epochstring = yr+'-'+mo+'-'+dy+' '+hr+':'+mn+':'+sec+' UTC'   

    #yr = str(year)
    #mo = str(month)
    if cdateStartEpoch.month() < 10:
        month = "0"+month
    #dy = str(day)
    if cdateStartEpoch.day() < 10:
        day = "0"+day
    datestring = year+"-"+month+"-"+day
    return datestring

def find_UT_hour(starttime, time):  #  given a time epoch (monte), finds the hour given a start time (Time obj def in userfile).  Ex: For 0 to 1, hour is 0;  for 23 to 24, hour is 23
    start_hour = starttime.hour
    start_min = starttime.minute
    start_sec = starttime.second
    secs_past_hr = start_min * 60 + start_sec

    if secs_past_hr == 0:
        secs_left_in_hr = 0.0
    else:
        secs_left_in_hr = 3600.0 - secs_past_hr

    print("secs_left_in_hr: ", secs_left_in_hr)

    start_epoch = covert_Time_to_Epoch(starttime)
    #tframe = start_epoch.frame()
    timedif = time.secondsPast( start_epoch ) / units.sec # number of seconds past the start_epoch

    timedif_less_hrfrac = timedif - secs_left_in_hr # time dif from start of next hour

    if timedif_less_hrfrac < 0.0:  # if time has not entered into the next hour
        hour  = start_hour

    else:  # if past the next hour, find which hour

        hrdif = int(timedif_less_hrfrac / 3600)
        print("hrdif: ", hrdif)
        hour = start_hour + hrdif   
        print("raw hour:" , hour)

        if hour > 23:  # it is into the next day so use the mod function
            hour = math.fmod(hour,24)
            print("hr if > 23: ", hour)

    print("seconds past start_epoch: ", timedif)
    print("number of seconds past the start_epoch ", timedif)
    print("time dif from start of next hour: ", timedif_less_hrfrac)
    print("hour: ", hour)
    
    return hour

class Time:  # takes a dict input with the fields below (contents are integers except for second, a float)

    def __init__(self,time):
        self.year = time['year']
        self.month = time['month']
        self.day = time['day']
        self.hour = time['hour']
        self.minute = time['minute']
        self.second = time['second']
        #self.julian = jultime(self.hour,self.minute,self.second,self.day,self.month,self.year)
        #self.gstdeg = gstdeg(self.julian)

    def displaytime(self,time):
        yr = str(time.year)
        mo = str(time.month)
        dy = str(time.day)
        hr = str(time.hour)
        mn = str(time.minute)
        sc = str(time.second)

        if time.minute < 10:
            minzero = "0"
        else:
            minzero = "" # use to add extra leading zeros if necessary
        if time.second < 10:
            seczero = "0"
        else:
            seczero = "" # use to add extra leading zeros if necessary

        self.stringout = hr+":"+minzero+mn+":"+seczero+sc+"  "+mo+"/"+dy+"/"+yr


if __name__ == "__main__":

    #time start:

    TIMESTART = {'year':2023,
            'month':3,
            'day':20,
            'hour':14,
            'minute':0,
            'second':0.0,
        }

    TIMEFINISH = {'year':2023,
            'month':3,
            'day':21,
            'hour':1,
            'minute':59,
            'second':59.0,
            }

    start=Time(TIMESTART)
    #start.displaytime(TIMESTART)

    finish=Time(TIMEFINISH)

    print("Start Epoch")
    startepoch = covert_Time_to_Epoch(start)
  
    #epoch:
    print("End Epoch")
    endepoch = covert_Time_to_Epoch(finish)

    #find the UT hour:
    
    hr = find_UT_hour(start, endepoch)

  






