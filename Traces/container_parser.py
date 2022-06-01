FILE = "container_usage_sub.csv"
"""
container_usage.csv format:
+-----------------------------------------------------------------------------------------+
| container_id     | string     |       | uid of a container                              |
| machine_id       | string     |       | uid of container's host machine                 |
| time_stamp       | double     |       | time stamp, in second                           |
| cpu_util_percent | bigint     |       |                                                 |
| mem_util_percent | bigint     |       |                                                 |
| cpi              | double     |       |                                                 |
| mem_gps          | double     |       | normalized memory bandwidth, [0, 100]           |
| mpki             | bigint     |       |                                                 |
| net_in           | double     |       | normarlized in coming network traffic, [0, 100] |
| net_out          | double     |       | normarlized out going network traffic, [0, 100] |
| disk_io_percent  | double     |       | [0, 100], abnormal values are of -1 or 101      |
+-----------------------------------------------------------------------------------------+
"""

total = {}
count = {}
ids = set()
max_size = 30000

with open(FILE,'r') as infile:
    for line in infile:
        #Currently want to average the utilization over the course of the given second
        container_id, machine_id, time_stamp, cpu_util_percent, mem_util_percent, cpi, mem_gps, mkpi, net_in, net_out, disk_io_percent = line.strip().split(",")
        time_stamp = float(time_stamp)
        if disk_io_percent:
            ids.add(container_id)
            if (len(ids)) > max_size:
                break
            disk_io_percent = float(disk_io_percent)
            if disk_io_percent < 0 or disk_io_percent > 100: 
                continue
            if time_stamp not in total:
                total[time_stamp] = disk_io_percent
                count[time_stamp] = 1
            else:
                total[time_stamp] += disk_io_percent
                count[time_stamp] += 1

outfile = open("ali_container_usage.dat", 'w')
out_list =[]
for key,v in sorted(total.items()):
    out_list.append(total[key]/count[key])
print(",".join(map(str,out_list)), file=outfile)
outfile.close()
