
def read_network(network):

    edge_num = 0

    node_num = 0

    data_lines = open(network, 'r').readlines()
    for data_line in data_lines[0:]:
        start, end = data_line.split()
        if int(start) < int(end):
            current = int(end)
        if node_num < current:
            node_num = current
        edge_num += 1

    node_num += 1


    indegree = [0] * node_num

    net = []

    for i in range(node_num):
        net.append([])

    data_lines = open(network, 'r').readlines()
    for data_line in data_lines[0:]:
        start, end = data_line.split()
        indegree[int(end)] += 1
        net[int(start)].append(int(end))
        net[int(start)].append(0.0)

    for i in range(node_num):
        for j in range(1, len(net[i]), 2):
            net[i][j] = 1/indegree[net[i][j-1]]

    return node_num, net



if __name__ == "__main__":

#
    for file in ['Brightkite_edges', 'F11_edges']:

        file_name = './' + file + '.txt'

        result = file.split('_')[0]

        node_num, net = read_network(file_name)

        with open('./' + result + '.txt', 'w') as f:

            for i in range(node_num):
                for j in range(1, len(net[i]), 2):
                    f.write(str(i) + '\t' + str(net[i][j-1]) + '\t' + str(net[i][j]) + '\n')







