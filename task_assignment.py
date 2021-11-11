import re
import time
import math
from math import sin, asin, cos, radians, fabs, sqrt, pow
import multiprocessing as mp
import random
from category_affinity import compute_lda, get_cat_affi

EARTH_RADIUS = 6371
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




taskNumber = 1500
workerNumber = 1200
valid_time = 5.0
MAXW = 1
RADIUS = 25
worker_speed = 5.0

lda_model, dictionary, worker_cat, worker_with_cat, location_venues = compute_lda()

file = './check-ins.txt'
task_time = '20110101'

records, execRec = [], {}

f = open(file, 'r')

time1 = time.time()

for line in f.readlines():

    data_line = line.strip().split('\t')

    if data_line[1][0:8] == task_time:
        records.append(data_line)

    user, task = int(data_line[0]), int(data_line[4])

    if user not in execRec.keys():
        execRec[user] = set()
    execRec[user].add(task)
f.close()

userId = []
taskId = []
userLoc, taskLoc, taskTime, taskCat = {}, {}, {}, {}
for r in records:
    u, t = map(int, [r[0], r[4]])
    if worker_with_cat[u] != 0:
        if u not in userLoc.keys():
            userLoc[u] = list(map(float, [r[2], r[3]]))
        userId.append(int(r[0]))

    if r[5] in location_venues.keys() and location_venues[r[5]] != 'NULL':

        if t not in taskLoc.keys():
            taskLoc[t] = list(map(float, [r[2], r[3]]))
            taskTime[t] = sum([a * b for a, b in zip([3600, 60, 1], map(int, [r[1][8:10], r[1][10:12], r[1][12:14]]))])
            taskCat[t] = location_venues[r[5]]

        taskId.append(int(r[4]))

userId = sorted(set(userId))
taskId = sorted(set(taskId))

workerlist = set(random.sample(userId, workerNumber))
tasklist = set(random.sample(taskId, taskNumber))


edges = []

class pGraph():

    def __init__(self):
        self.network = dict()

    def add_node(self, node):
        if node not in self.network:
            self.network[node] = dict()

    def add_edge(self, s, e, w):

        pGraph.add_node(self, s)
        pGraph.add_node(self, e)
        self.network[s][e] = w

    def get_out_degree(self, source):
        return len(self.network[source])

    def get_neighbors(self, source):
        return self.network[source].items()

    def get_node(self, node):
        if node not in self.network:
            self.network[node] = None

class Graph:

    def __init__(self):
        self.network = dict()

    def add_node(self, node):
        if node not in self.network:
            self.network[node] = dict()

    def add_edge(self, s, e, w):

        Graph.add_node(self, s)
        Graph.add_node(self, e)

        self.network[e][s] = w

    def get_out_degree(self, source):
        return len(self.network[source])

    def get_neighbors(self, source):
        if source in self.network:
            return self.network[source].items()
        else:
            return []

    def get_neighbors_keys(self, source):
        if source in self.network:
            return self.network[source].keys()
        else:
            return []

class Worker(mp.Process):
    def __init__(self, inQ, outQ):
        super(Worker, self).__init__(target=self.start)
        self.graph = graph
        self.availableNode = availableNode
        self.node_num = node_num
        self.inQ = inQ
        self.outQ = outQ
        self.R = []
        self.count = 0

    def run(self):

        while True:
            theta = self.inQ.get()

            while self.count < theta:

                v = random.sample(self.availableNode, 1)[0]

                rr = generate_rr(v, self.graph)

                self.R.append(rr)
                self.count += 1
            self.count = 0
            self.outQ.put(self.R)
            self.R = []

def create_worker(num):

    global worker
    for i in range(num):

        worker.append(Worker(mp.Queue(), mp.Queue()))
        worker[i].start()

def finish_worker():

    for w in worker:
        w.terminate()

def sampling(epsoid, l):
    global graph, seed_size, worker, availableNode
    R = []
    LB = 1
    n = node_num
    k = seed_size
    epsoid_p = epsoid * math.sqrt(2)
    worker_num = 1
    create_worker(worker_num)
    for i in range(1, int(math.log2(n - 1)) + 1):

        x = n / (math.pow(2, i))
        lambda_p = ((2 + 2 * epsoid_p / 3) * (logcnk(n, k) + l * math.log(n) + math.log(math.log2(n))) * n) / pow(
            epsoid_p, 2)
        theta = lambda_p / x

        for ii in range(worker_num):
            worker[ii].inQ.put((theta - len(R)) / worker_num)
        for w in worker:
            R_list = w.outQ.get()
            R += R_list


        Si, f = node_selection(R, k)

        if n * f >= (1 + epsoid_p) * x:
            LB = n * f / (1 + epsoid_p)
            break

    alpha = math.sqrt(l * math.log(n) + math.log(2))
    beta = math.sqrt((1 - 1 / math.e) * (logcnk(n, k) + l * math.log(n) + math.log(2)))
    lambda_aster = 2 * n * pow(((1 - 1 / math.e) * alpha + beta), 2) * pow(epsoid, -2)
    theta = lambda_aster / LB
    length_r = len(R)
    diff = theta - length_r

    if diff > 0:

        for ii in range(worker_num):
            worker[ii].inQ.put(diff / worker_num)
        for w in worker:
            R_list = w.outQ.get()
            R += R_list

    finish_worker()
    return R

def generate_rr(v, graph):

    return generate_rr_ic(v, graph)

def node_selection(R, k):
    Sk = set()
    rr_degree = [0 for ii in range(node_num + 1)]
    node_rr_set = dict()

    matched_count = 0
    for j in range(0, len(R)):
        rr = R[j]
        for rr_node in rr:
            rr_degree[rr_node] += 1
            if rr_node not in node_rr_set:
                node_rr_set[rr_node] = list()
            node_rr_set[rr_node].append(j)

    for i in range(k):
        max_point = rr_degree.index(max(rr_degree))
        Sk.add(max_point)
        matched_count += len(node_rr_set[max_point])
        index_set = []
        for node_rr in node_rr_set[max_point]:
            index_set.append(node_rr)
        for jj in index_set:
            rr = R[jj]
            for rr_node in rr:
                rr_degree[rr_node] -= 1
                node_rr_set[rr_node].remove(jj)
    return Sk, matched_count / len(R)

def generate_rr_ic(node, graph):
    activity_set = list()
    activity_set.append(node)
    activity_nodes = list()
    activity_nodes.append(node)
    while activity_set:
        new_activity_set = list()
        for seed in activity_set:
            for node, weight in graph.get_neighbors(seed):
                if node not in activity_nodes:
                    if random.random() <= weight:
                        activity_nodes.append(node)
                        new_activity_set.append(node)
        activity_set = new_activity_set

    return activity_nodes

def influence_calculation(R):

    influence = [0.0] * node_num

    rr_degree = [0 for ii in range(node_num + 1)]
    node_rr_set = dict()

    for j in range(0, len(R)):
        rr = R[j]

        for rr_node in rr:
            rr_degree[rr_node] += 1
            if rr_node not in node_rr_set:
                node_rr_set[rr_node] = list()

            node_rr_set[rr_node].append(j)

    for i in range(node_num):
        if i not in node_rr_set:
            influence[i] = 0.0
        else:
            influence[i] = node_num*len(node_rr_set[i])/len(R)

    return influence

def rrpo(epsoid, l):

    n = node_num
    l = l * (1 + math.log(2) / math.log(n))
    R = sampling(epsoid, l)
    return R

def logcnk(n, k):
    res = 0
    for i in range(n - k + 1, n + 1):
        res += math.log(i)
    for i in range(1, k + 1):
        res -= math.log(i)
    return res

def read_file(network):

    edge_num = 0

    node_num = 0

    data_lines = open(network, 'r').readlines()
    for data_line in data_lines[0:]:
        start, end, weight = data_line.split()
        if int(start) < int(end):
            current = int(end)
        if node_num < current:
            node_num = current
        edge_num += 1
        graph.add_edge(int(start), int(end), float(weight))
        pgraph.add_edge(int(start), int(end), float(weight))
        availableNode.append(int(start))
        availableNode.append(int(end))

    node_num += 1


    return node_num, edge_num

graph = Graph()
pgraph = pGraph()

def getPareto(node_num):

    file = open("./pareto_and_randomwalk.txt")
    data = []
    for line in file.readlines():
        data.append(line.strip())

    file.close()
    b = []
    for row in range(node_num):
        b.append([])

    for i in range(len(data)):
        c = re.split('\t', data[i])
        b[int(c[0])].append(int(c[0]))
        b[int(c[0])].append(float(c[1]))
        b[int(c[0])].append(float(c[2]))
        b[int(c[0])].append(float(c[3]))
        b[int(c[0])].append(float(c[4]))
    return b

def nodeToTask(b, node, target_lat, target_lng):

    count = 0.0

    if b[node] == []:
        return 0.0

    pareto = b[node][1]

    for j in range(0, len(b[node]), 5):
        count += b[node][j + 4] * pow(get_distance_hav(b[node][j + 2], b[node][j + 3], target_lat, target_lng) + 1.0,
                                       -pareto)

    return count



if __name__ == "__main__":

    node_num = 0
    edge_num = 0
    degree = []

    availableNode = []

    start = time.process_time()


    network = './F11.txt'

    if network == './Brightkite.txt':
        id = 'B'
    elif network == './F11.txt':
        id = 'F11'
    else:
        raise IOError('no such dataset')

    node_num, edge_num = read_file(network)

    availableNode = set(availableNode)

    worker = []
    epsoid = 0.1
    l = 1

    start1 = time.time()

    seed_size = 1

    R = rrpo(epsoid, l)

    R_inf = {}

    for j in range(0, len(R)):

        node = R[j][0]

        if node not in R_inf.keys():
            R_inf[node] = list()
            R_inf[node] += R[j]
        else:
            R_inf[node] += R[j]

    b = getPareto(node_num)

    for u in workerlist:

        worker_vec = dictionary.doc2bow(worker_cat[u])
        score_w = [0.0] * 50

        for index, score in lda_model[worker_vec]:
            score_w[index] = score

        inf = [0.0] * node_num

        for i in range(node_num):
            if i not in R_inf.keys():
                continue
            else:
                rr = R_inf[i]
                inf[i] = rr.count(u)*node_num / len(R)

        for t in tasklist:


            if get_distance_hav(*(taskLoc[t]+userLoc[u])) > RADIUS:
                continue

            if get_distance_hav(*(taskLoc[t]+userLoc[u])) / worker_speed > valid_time:
                continue


            weight = 0.0
            score_t = [0.0] * 50
            task_vec = dictionary.doc2bow(taskCat[t])
            for index, score in lda_model[task_vec]:
                    score_t[index] = score

            for item in range(len(score_t)):
                weight += score_w[item]*score_t[item]


            target_lat = taskLoc[t][0]
            target_lng = taskLoc[t][1]

            count = 0.0
            inf_threshold = 1e-3
            for j in range(node_num):
                if inf[j] <= inf_threshold:
                    continue
                else:
                    count += inf[j] * nodeToTask(b, j, target_lat, target_lng)


            cost = 1.0 / (weight * count + 1)

            edges.append([u, t, 1, cost])

    n, m = len(userId)+len(taskId)+2, len(userId)+len(edges)+len(taskId)
    s, t = n-1, n
    li = []
    for i in range(1, len(userId)+1):
        li.append([s, i, 1, 0])
    for e in edges:
        user = userId.index(e[0]) + 1
        task = taskId.index(e[1]) + 1 + len(userId)
        li.append([user, task, 1, float(e[3]*100)])

    for i in range(len(userId)+1, s):
        li.append([i, t, MAXW, 0])

    outputFile = './' + str(taskNumber) + '_' + str(workerNumber) + '_' + str(valid_time) + '_' + str(RADIUS) + '_in.txt'

    with open(outputFile, 'w') as f:
        f.write(' '.join(map(str, [n, m, s, t])) + '\n')
        for l in li:
            f.write(' '.join(map(str, l)) + '\n')


