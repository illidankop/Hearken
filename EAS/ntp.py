import subprocess
import time

cmd = "ntpdate -u 192.168.6.10"
for i in range(10000):
    subprocess.call(cmd, shell=True)
