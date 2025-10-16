import numpy as np
from utils.log_print import *

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# this function recieves a list of numbers that represent evenly spaced azimuth measurements
# and checks if the list matches the following conditions:
# the list is either in ascending or descending order. if not, return False
# the difference between the first and last element is less than 180 degrees
#the difference between the first and last element is more than 30 degrees
# no more than 2 values have a difference of more than 3 times the standard deviation of the difference
def is_valid_slope(azimuth_list, blasts_aoa, min_aoa_4_check_slope):
    try:
        max_diff_in_deg = 10
        # validate that azimuth list size is at least the size of min_aoa_4_check_slope
        if len(azimuth_list) < min_aoa_4_check_slope:
            logPrint( "INFO", E_LogPrint.LOG, f"is_valid_slope - less than {min_aoa_4_check_slope} azimuths (1sec) in slope input")
            return False, None
            
        all_filtered_lists = []
        filtered_list = []
        inconsistent_flag = False
        inconsistent_data = None
        # build aoas lists ,two adjacent inconsistecy of any kind are not alowd
        for i, az in enumerate(azimuth_list):
            # find greater and smaller azimuth in current cycle, for easier comparision
            inconsistent_flag, inconsistent_data, is_red_flag = \
                create_or_extend_azimuth_list(azimuth_list, max_diff_in_deg, all_filtered_lists, filtered_list,
                                            inconsistent_flag, inconsistent_data, i, az, is_last_chance=False)
            if is_red_flag:                
                inconsistent_flag, inconsistent_data, is_red_flag = \
                    create_or_extend_azimuth_list(azimuth_list, max_diff_in_deg, all_filtered_lists, filtered_list, 
                                                True, (i, az), i, az, is_last_chance=True)

        all_filtered_lists.append(filtered_list.copy())
        
        # logPrint( "INFO", E_LogPrint.LOG, f"all azimuth lists: {all_filtered_lists}")
        longest_list = max(all_filtered_lists,key=len)
        azimuth_list = [a[-1] for a in longest_list]
        if len(azimuth_list) < min_aoa_4_check_slope:
            logPrint( "INFO", E_LogPrint.BOTH, f"is_valid_slope - longest azimuth list {azimuth_list} contains {len(azimuth_list)} aoas of min {min_aoa_4_check_slope} required")
            return False, None
        
        # There is at least one long enough azimuth tuple (index, aoa) list candidate to keep working 
        over_length_az_lists = [az_lst for az_lst in all_filtered_lists if len(az_lst) > min_aoa_4_check_slope]
        # TODO: should be sorted by distance from 1st to last aoa instead of size
        over_length_az_lists = sorted([az_lst for az_lst in over_length_az_lists], key=lambda x: len(x), reverse=True)     
        for one_az_tuple in over_length_az_lists:
            azimuth_list = [val[1] for val in one_az_tuple]
            logPrint( "INFO", E_LogPrint.LOG, f"check following azimuth lists: {azimuth_list}")

            if blasts_aoa > azimuth_list[-2] and azimuth_list[-1] > azimuth_list[-2] or \
                blasts_aoa < azimuth_list[-2] and azimuth_list[-1] < azimuth_list[-2]:
                logPrint( "INFO", E_LogPrint.LOG, f"is_valid_slope - azimuth list {azimuth_list} generate discontinuity slope comparing to blast's aoa {blasts_aoa}")
                continue

            # check if the difference between the first and last element is less than 180 degrees
            if azimuth_list[0] >= 270 and azimuth_list[-1] <= 90:
                azimuth_list[-1] += 360
            elif azimuth_list[-1] >= 270 and azimuth_list[0] <= 90:
                azimuth_list[0] += 360

            if np.abs(azimuth_list[0] - azimuth_list[-1]) > 190:
                print(f'the difference between the first and last element is more than 180 degrees - {np.abs(azimuth_list[0] - azimuth_list[-1])}')
                logPrint( "INFO", E_LogPrint.LOG, f'is_valid_slope - the difference between the first and last element is more than 180 degrees - {np.abs(azimuth_list[0] - azimuth_list[-1])}')
                continue

            # check that the difference between the first and last element is more than 20 degrees
            mav_5 = moving_average(azimuth_list, 5)
            az_diff = max(mav_5) - min(mav_5)
            if max(mav_5) - min(mav_5) < 20:
                # print(f'the difference between the max and min moving average(5) is {az_diff} less than 20 degrees required')
                logPrint( "INFO", E_LogPrint.BOTH, f'the difference between the max and min moving average(5) is {az_diff} less than 20 degrees required')
                continue

            return True, one_az_tuple
    except Exception as ex:
        logPrint("ERROR", E_LogPrint.BOTH, f"is valis_slope following exception was cought: {ex}")

    return False, None

def create_or_extend_azimuth_list(azimuth_list, max_diff_in_deg, all_filtered_lists, filtered_list, inconsistent_flag, inconsistent_data, i, az, is_last_chance):        
    is_red_flag = False
    if(i > 0):
        last_used_idx = i-1 if inconsistent_flag == False or is_last_chance else i-2
        greater = az if az > azimuth_list[last_used_idx] else azimuth_list[last_used_idx]
        smaller = az if az < azimuth_list[last_used_idx] else azimuth_list[last_used_idx]
                                    
            # when filtered list length >= 2 elements asc/desc inconsistency is being checked
            # taking in consideration cyclic condition
    is_check_asc_desc = False
    if len(filtered_list) >= 2:
        is_check_asc_desc = True
        prev_taken = filtered_list[-2][1]
        if (360 - prev_taken) < (2 * max_diff_in_deg):
            prev_taken = -1 * (360 - prev_taken)
                    
        last_taken = filtered_list[-1][1]
        if (360 - last_taken) < max_diff_in_deg:
            last_taken = -1 * (360 - last_taken)
                    
        cur_az = az
        if (360 - cur_az) < max_diff_in_deg:
            cur_az = -1 * (360 - cur_az)
                        
            # 1st element is always OK
    if len(filtered_list) == 0:
        inconsistent_flag = False
        filtered_list.append((i, az))
                    
            # if aoa diff between adjacent measurements is greater than allowed, ignore current measure and mark as inconsistent
            # if inconsistent_flag (yellow flag) is allready set to true, we store current list and start building new list from current angle
    elif ((greater - smaller > max_diff_in_deg) and abs(360-greater) + smaller > max_diff_in_deg):
                # taken aoa are in consistent order by now, set inconsistent flag to true
        if not inconsistent_flag:
            inconsistent_data = (i, az)
            inconsistent_flag = True
                                
                # inconsistent flag is already turned on should rebuild list starting with current aoa
        else:                    
            all_filtered_lists.append(filtered_list.copy())
            filtered_list.clear()                        
            filtered_list.append((inconsistent_data[0], inconsistent_data[1]))
            is_red_flag = True
            inconsistent_flag = False
            inconsistent_data = None
            
            # aoa diff between adjacent measurements is OK keep building current azimuth list
    elif len(filtered_list) < 2:
        inconsistent_flag = False
        filtered_list.append((i, az))

            # check if angles represents ascending/descending consistency        
    elif is_check_asc_desc and ((cur_az > last_taken > prev_taken) or (cur_az < last_taken < prev_taken)):
        inconsistent_flag = False
        filtered_list.append((i, az))
            
            # ascending/descending inconsistency                    
    elif not inconsistent_flag:
        inconsistent_data = (i, az)
        inconsistent_flag = True        

            # rebuild aoa list starting with last stored angle
    else:
        all_filtered_lists.append(filtered_list.copy())
        filtered_list.clear()        
        filtered_list.append((inconsistent_data[0], inconsistent_data[1]))        
        is_red_flag = True
        inconsistent_flag = False
        inconsistent_data = None
    
    return inconsistent_flag, inconsistent_data, is_red_flag            
        

# def is_valid_slope(azimuth_list, blasts_aoa, min_aoa_4_check_slope):
#     max_diff_in_deg = 10
#     # check that the list is not empty
#     if len(azimuth_list) < min_aoa_4_check_slope:
#         logPrint( "INFO", E_LogPrint.LOG, f"is_valid_slope - less than {min_aoa_4_check_slope} azimuths (1sec) in slope input")
#         return False, None
    
#     orig_len = len(azimuth_list)
#     filtered_list = []
#     inconsistent_flag = False
    
#     # build aoas list , two adjacent inconsistecy of any kind are not alowd
#     for i, az in enumerate(azimuth_list):        
#         if(i > 0):
#             last_used_idx = i-1 if inconsistent_flag == False else i-2
#             greater = az if az > azimuth_list[last_used_idx] else azimuth_list[last_used_idx]
#             smaller = az if az < azimuth_list[last_used_idx] else azimuth_list[last_used_idx]
        
#         is_check_asc_desc = False
#         if len(filtered_list) >= 2:
#             is_check_asc_desc = True
#             prev_taken = filtered_list[-2][1]
#             if (360 - prev_taken) < (2 * max_diff_in_deg):
#                 prev_taken = -1 * (360 - prev_taken)
                
#             last_taken = filtered_list[-1][1]
#             if (360 - last_taken) < max_diff_in_deg:
#                 last_taken = -1 * (360 - last_taken)
                
#             cur_az = az
#             if (360 - cur_az) < max_diff_in_deg:
#                 cur_az = -1 * (360 - cur_az)
            

#         # 1st element
#         if len(filtered_list) == 0:
#             inconsistent_flag = False
#             filtered_list.append((i, az))
                
#         # aoa diff between adjacent measurements is greater than allowed, ignore current measure and mark as inconsistent
#         elif ((greater - smaller > max_diff_in_deg) and abs(360-greater) + smaller > max_diff_in_deg):
#             # taken aoa are in consistent order by now, set inconsistent flag to true
#             if not inconsistent_flag:
#                 inconsistent_flag = True
#             # inconsistent flag is already turned on should rebuild list starting with current aoa
#             else:
#                 filtered_list.clear()
#                 inconsistent_flag = False            
#                 filtered_list.append((i, az))
                
#         elif len(filtered_list) < 2:
#             inconsistent_flag = False            
#             filtered_list.append((i, az))

#         # check if angles represents ascending/descending consistency        
#         elif is_check_asc_desc and ((cur_az > last_taken > prev_taken) or (cur_az < last_taken < prev_taken)):
#             inconsistent_flag = False
#             filtered_list.append((i, az))
        
#         # ascending/descending inconsistency                    
#         elif not inconsistent_flag:
#             inconsistent_flag = True
#         # rebuild aoa list starting with last stored angle
#         else:
#             inconsistent_flag = False                    
#             filtered_list.clear()
#             filtered_list.append((i-1, azimuth_list[i-1]))
#             filtered_list.append((i, az))            

#     azimuth_list = [a[-1] for a in filtered_list]
#     if len(azimuth_list) < min_aoa_4_check_slope:
#         logPrint( "INFO", E_LogPrint.BOTH, f"is_valid_slope - filtered list {azimuth_list} contains less than {min_aoa_4_check_slope} azimuths (1sec)")
#         return False, None

#     # remove values which are more than 10 degrees away from the previous value and the next value
#     azimuth_tuple = [(i, x) for i, x in enumerate(azimuth_list) if i == 0 or i == len(azimuth_list)-1 or (np.abs(x - azimuth_list[i+1]) > 350 or np.abs(x - azimuth_list[i-1]) > 350 or np.abs(x - azimuth_list[i-1]) < 10 or np.abs(x - azimuth_list[i+1]) < 10)]    
#     azimuth_list = [val[1] for val in azimuth_tuple]
    
#     if blasts_aoa > azimuth_list[-2] and azimuth_list[-1] > azimuth_list[-2] or \
#         blasts_aoa < azimuth_list[-2] and azimuth_list[-1] < azimuth_list[-2]:
#         logPrint( "INFO", E_LogPrint.LOG, f"is_valid_slope - discontinuity slope")
#         return False, None
    
#     jumps = [x for x in np.diff(azimuth_list) if abs(x) > 10]
#     outliers_cnt = orig_len - len(azimuth_list) + len(jumps)
#     if outliers_cnt/orig_len > 0.2:
#         print(f'too many outliers ({outliers_cnt})')
#         logPrint( "INFO", E_LogPrint.LOG, f'is_valid_slope - too many outliers ({outliers_cnt})')
#         return False, None
    
#     # check if most of the values are ascending or descending
#     val_diffs = np.diff(azimuth_list)
#     if np.max(val_diffs) < -180 or np.min(val_diffs) > 180:
#         val_diffs = np.diff(np.mod((azimuth_list+180),360))

#     # check if the difference between the first and last element is less than 180 degrees
#     if azimuth_list[0] >= 270 and azimuth_list[-1] <= 90:
#         azimuth_list[-1] += 360
#     elif azimuth_list[-1] >= 270 and azimuth_list[0] <= 90:
#         azimuth_list[0] += 360

#     if np.abs(azimuth_list[0] - azimuth_list[-1]) > 190:
#         print(f'the difference between the first and last element is more than 180 degrees - {np.abs(azimuth_list[0] - azimuth_list[-1])}')
#         logPrint( "INFO", E_LogPrint.LOG, f'is_valid_slope - the difference between the first and last element is more than 180 degrees - {np.abs(azimuth_list[0] - azimuth_list[-1])}')
#         return False, None
    
#     # check that the difference between the first and last element is more than 30 degrees
#     mav_5 = moving_average(azimuth_list, 5)
#     if max(mav_5) - min(mav_5) < 20:
#         print(f'the difference between the max and min moving average(5) is less than 20 degrees - {max(mav_5) - min(mav_5)}')
#         logPrint( "INFO", E_LogPrint.LOG, f'is_valid_slope - the difference between the max and min moving average(5) is less than 20 degrees - {max(mav_5) - min(mav_5)}')
#         return False, None
    
#     # if np.abs(azimuth_list[0] - azimuth_list[-1]) < 20:
#     #     print(f'the difference between the first and last element is less than 30 degrees - {np.abs(azimuth_list[0] - azimuth_list[-1])}')
#     #     return False
    
#     # check if more than 80% of the values are ascending
#     if np.sum(np.where(val_diffs > 0)) > int(len(val_diffs)*0.8):
#         asc = True
#     # check if more than 80% of the values are descending
#     elif np.sum(np.where(val_diffs < 0)) > int(len(val_diffs)*0.8):
#         dsc = True
#     else:
#         print('not more than 80% of the values are ascending or descending')
#         logPrint( "INFO", E_LogPrint.LOG, f'is_valid_slope - not more than 80% of the values are ascending or descending')
#         return False, None

#     return True, azimuth_tuple


# helper function to load a list of azimuth values from the second column of a csv file
def load_azimuths(filename):
    return np.loadtxt(filename, delimiter=',', usecols=1, skiprows=1)


# main function to test the is_valid_slope function           
if __name__ == '__main__':
    import glob
    import os
    post_event_az = [275.99999999999994, 280.0, 282.0, 284.0, 286.0, 288.0, 286.0, 286.0, 284.0, 284.0, 282.0, 290.00000000000006]
    post_event_az = [275.99999999999994, 280.0, 282.0, 284.0, 286.0, 288.0, 286.0, 286.0, 284.0, 284.0, 282.0, 290.00000000000006, 292.0, 296.0, 296.0, 292.0, 306.0, 314.0, 318.0, 314.0, 322.00000000000006, 336.0, 338.0, 342.0, 318.0, 196.0, 196.0, 190.0, 174.0, 168.0, 164.0, 158.0]
    a = is_valid_slope(post_event_az,240, 12)
    pass
    
    files_dir = 'D:\\recordings\\forEli'


    # loop through the csv files
    for r, d, f in os.walk(files_dir):
        for file in f:
            # check that the file ends with .csv
            if not file.endswith(".csv") or not file.__contains__('_f_'):
                continue

            # load the azimuth values from the csv file
            azimuths = load_azimuths(r+'\\' + file)
            # check if the slope is valid
            if is_valid_slope(azimuths[1:30]):
                print(f'{file} : valid')
            else:
                print(f'{file} : not valid')

# changed by Erez in version 3.1.0:
    # improvements and bug fix of is_valid_slope function
# changed by gonen in version 3.2.3 (ATM-merged):
    # is_valid_slope return also time-azimuth tupple
# changed by gonen in version 3.2.8:
    # change is_valid_slope signiture add blasts's aoa and min_aoa_4_check_slope
    # in is_valid_slope, build aoas lists instead of removing mismatch aoas and test all lists for check slope
    