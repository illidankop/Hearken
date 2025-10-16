import time

from pysine import play_data, get_sine


# measure the smallest time delta by spinning until the time changes
def next1sec():
    t0 = time.time_ns()
    next_sec = (t0 // 1000000000 + 1) * 1000000000
    t1 = t0
    while t1 < next_sec:
        t1 = time.time_ns()
    return t1


# measure the smallest time delta by spinning until the time changes
def next_n_sec(seconds=1):
    t0 = time.time_ns()
    rem = seconds - (t0 // 1000000000 + seconds) % seconds
    next_sec = ((t0 // 1000000000) + rem) * 1000000000
    t1 = t0
    while t1 < next_sec:
        t1 = time.time_ns()
    return t1


# measure the smallest time delta by spinning until the time changes
def timer():
    t0 = time.time_ns()
    t1 = t0
    while t1 == t0:
        t1 = time.time_ns()
    return (t0, t1, t1 - t0)


# samples = [measure() for i in range(100)]
#
# for s in samples:
#     print (s)

for i in range(100):

    frq = 5000
    jump = 300
    jump_count = 5
    duration = 0.1

    audio = None
    for i in range(jump_count):
        d = get_sine(frq + (i * jump), duration)
        audio = audio + bytearray(d) if audio is not None else bytearray(d)

    # audio = bytearray(audio_data) + bytearray(chirp_data)

    audio_data2 = get_sine(frequency=frq + 200, duration=0.1)
    audio_data3 = get_sine(frequency=frq + 400, duration=0.1)
    audio_data4 = get_sine(frequency=frq + 600, duration=0.1)
    audio_data5 = get_sine(frequency=frq + 800, duration=0.1)
    stair = audio + bytearray(audio_data2) + bytearray(audio_data3) + bytearray(
        audio_data4) + bytearray(audio_data5)
    sent_time = next_n_sec(5)
    # sine(frequency=frq, duration=1)  # plays a 1s sine wave at 4400 Hz
    play_data(bytes(audio))
    time_after = time.time_ns()
    print(f'sent: sent_time {sent_time} After time: {time_after}')
    with open('send_times.txt', 'a+') as fd:
        fd.write(f'{sent_time} \n')

# import sched

# def myfunc():
#     print(time.time_ns())
#
#
# scheduler = sched.scheduler(time.time, time.sleep)
# t = time.strptime('2020-08-18 09:21:00', '%Y-%m-%d %H:%M:%S')
# t = time.mktime(t)
# scheduler_e = scheduler.enterabs(t, 1, myfunc, ())
# scheduler.run()
#
#
# time.sleep(80)
