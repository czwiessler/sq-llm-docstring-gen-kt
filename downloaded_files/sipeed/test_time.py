from maix import time

t = time.time()
t2 = time.ticks_s()
print("time:", t)
print("time from bootup:", t2)
print("time_ms:", time.time_ms())
print("time_us:", time.time_us())

time.sleep(1)
time.sleep_ms(200)

print("time_diff from bootup:", time.ticks_diff(t2))

datetime = time.gmtime(t)
print(datetime.strftime("%Y-%m-%d %H:%M:%S %z"), datetime.timestamp(), t)
datetime = time.now()
print(datetime.strftime("%Y-%m-%d %H:%M:%S %z"), datetime.timestamp())

datetime = time.localtime()
print(datetime.strftime("%Y-%m-%d %H:%M:%S %z"), datetime.timestamp())

datetime = time.strptime("2023-03-19 00:00:00", "%Y-%m-%d %H:%M:%S")
print(datetime.strftime("%Y-%m-%d %H:%M:%S %z"), datetime.timestamp())


