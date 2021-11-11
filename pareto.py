import re
from numpy import *
from datetime import datetime
from math import sin, asin, cos, radians, fabs, sqrt

EARTH_RADIUS = 6371
N = 11326

def hav(theta):
    s = sin(theta / 2)
    return s * s


def get_distance_hav(lat0, lng0, lat1, lng1):

    lat0 = radians(lat0)
    lat1 = radians(lat1)
    lng0 = radians(lng0)
    lng1 = radians(lng1)

    dlng = fabs(lng0 - lng1)
    dlat = fabs(lat0 - lat1)
    h = hav(dlat) + cos(lat0) * cos(lat1) * hav(dlng)
    distance = 2 * EARTH_RADIUS * asin(sqrt(h))

    return distance


def read_file_checkin():
    file = open("./checkins")

    data = []
    b = []
    for line in file.readlines():
        data.append(line.strip())
    file.close()

    for row in range(N):
        b.append([])

    for i in range(len(data)):
        if i == 0:
            continue
        c = re.split('\t', data[i])

        if(len(c) == 5):
            if(c[2]!='0.0' and c[3] != '0.0'):
                s = c[4]
                a = re.split('-|T|Z|:| ', s)
                s = ""
                for i in a:
                    s += i
                b[int(c[0][1:])].append(c[0][1:])
                b[int(c[0][1:])].append(s)
                b[int(c[0][1:])].append(c[2])
                b[int(c[0][1:])].append(c[3])
                b[int(c[0][1:])].append(c[1])

    return b

b = read_file_checkin()




fileb = open("./pareto.txt",'w+')

for ch in b:

    a = ch
    a1 = []
    for i in range(1, len(a), 5):
        a1.append(a[i:i + 4])

    a1.sort()

    i = 0

    while (i < len(a1)):

        ch1 = a1[i]

        location = ch1[1] + ',' + ch1[2]

        t1 = ch1[0]
        year1 = t1[0:4]
        month1 = t1[4:6]
        day1 = t1[6:8]
        hour1 = t1[8:10]
        minute1 = t1[10:12]
        second1 = t1[12:14]

        time1 = str(year1) + '-' + str(month1) + '-' + str(day1)

        date1 = datetime.strptime(time1, "%Y-%m-%d")

        j = i + 1
        while j < len(a1):
            ch2 = a1[j]
            t2 = ch2[0]
            location = ch2[1] + ',' + ch2[2]
            year2 = t2[0:4]
            month2 = t2[4:6]
            day2 = t2[6:8]
            hour2 = t2[8:10]
            minute2 = t2[10:12]
            second2 = t2[12:14]
            time2 = str(year2) + '-' + str(month2) + '-' + str(day2)

            date2 = datetime.strptime(time2, "%Y-%m-%d")

            if (date2 - date1).total_seconds() / (3600.0 * 24) <= 2:

                time = year1 + month1 + day1 + hour2 + minute2 + second2
                a1[j][0] = time
                j += 1
            else:
                break



        i = j
        if i >= len(a1):
            break


    lo = []
    for i in a1:

        lo.append(str(i[1])+ ', '+ str(i[2]))


    num = set(lo)

    num = list(num)

    dic = {num[i]: i for i in range(len(num))}

    for i in a1:

        i[3] = dic[str(i[1])+ ', '+ str(i[2])]

        fileb.write(str(a[0]) + '\t' + str(i[0]) + '\t' + str(i[1]) + '\t' + str(i[2]) + '\t' + str(i[3]) + '\n')#check_ins文件


fileb.close()

