from datetime import datetime


#TODO: replace with extension methods style, if possible 
#int
def date_time_to_sec_since1970_float(dt):
    return (dt-datetime(1970,1,1)).total_seconds()
def date_time_to_milisec_since1970_float(dt):
    return (dt-datetime(1970,1,1)).total_seconds() * 1e3
def date_time_to_microsec_since1970_float(dt):
    return (dt-datetime(1970,1,1)).total_seconds() * 1e6
#float
def date_time_to_sec_since1970_int(dt):
    return int((dt-datetime(1970,1,1)).total_seconds())    
def date_time_to_milisec_since1970_int(dt):
    return int((dt-datetime(1970,1,1)).total_seconds() * 1e3)
def date_time_to_microsec_since1970_int(dt):
    return int((dt-datetime(1970,1,1)).total_seconds() * 1e6)