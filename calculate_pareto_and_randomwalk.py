
from numpy import *
import numpy as np
import re
from itertools import chain
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

file = open("./pareto.txt")

data = []
b = []
for line in file.readlines():
    data.append(line.strip())
file.close()

for row in range(N):
    b.append([])

for i in range(len(data)):
    c = re.split('\t', data[i])

    b[int(c[0])].append(c[0])
    b[int(c[0])].append(c[1])
    b[int(c[0])].append(c[2])
    b[int(c[0])].append(c[3])
    b[int(c[0])].append(c[4])



prlist = []*N
pareto = [9999999.0]*N

for i in b:
    c = i
    if(c == []):

        continue
    lenmat = 0
    for i in range(0, len(c), 5):
        if (lenmat < int(c[i + 4])):
            lenmat = int(c[i + 4])


    A = np.mat(zeros((lenmat + 1, lenmat + 1)))

    for i in range(4, len(c) - 5, 5):

        if (int(c[i]) == int(c[i + 5])):
            continue
        else:
            A[int(c[i]), int(c[i + 5])] = 1

    count = A.sum(axis=1)

    for i in range(A.shape[0]):

        if (count[i] == 0):
            A[i, :] = 1 / A.shape[1]
        else:
            A[i, :] = A[i, :] / count[i]


    PR = np.mat(ones(A.shape[0]))

    PR = PR / PR.sum(axis=1)

    PR = PR.T

    s = 0.15
    restart = np.mat(ones((A.shape[0], A.shape[0])))

    A1 = (1 - s) * A + s / A.shape[0] * restart

    A1 = A1.T

    A2 = A1

    for i in range(1000):
        temp = A2
        A2 = A2 * A1
        if (abs(temp - A2).sum() < 0.001):
            break

    prnode = A2 * PR

    prlist.append(c[0])
    prlist.append(prnode.tolist())

    n = len(c)/5

    sum = 0.0
    for i in range(0,len(c)-5,5):
        lat1 = float(c[i+2])
        lon1 = float(c[i+3])
        lat2 = float(c[i+7])
        lon2 = float(c[i+8])
        dis = get_distance_hav(lat1, lon1, lat2, lon2)

        sum += math.log(dis+1)


    if (sum == 0.0):
        pareto[int(c[i])] = (9999999.0)
    else:

        pareto[int(c[0])] = (n - 1) / sum

nodeprlist = []


for i in range(0,len(prlist),2):

    j = int(prlist[i])
    pr1 = prlist[i+1]
    a = list(chain.from_iterable(pr1))
    nodeprlist.append(j)
    nodeprlist.append(a)

file = open("./pareto_and_randomwalk.txt",'w+')

for i in b:
    c = i
    if(c == []):
        continue
    else:

        l1 = nodeprlist.index(int(c[0]))

        for k in range(len(nodeprlist[l1+1])):
            h = 4
            while(h<len(c)):
                if(int(c[h]) == k):
                    file.write(c[0] + '\t' + str(pareto[int(c[0])]) + '\t' + str(c[h - 2]) + '\t' + str(c[h - 1]) + '\t' + str(nodeprlist[l1 + 1][k]) + '\n')
                    break
                h += 5

file.close()

