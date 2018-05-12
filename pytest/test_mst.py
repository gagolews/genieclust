#import networkx as nx
#import numpy as np

#def test_MST_list(mst, D):
    #G = nx.Graph()
    #for i in range(D.shape[0]-1):
        #for j in range(i+1, D.shape[0]):
            #G.add_edge(i, j, weight=D[i, j])

    #mst_test = []
    #G = nx.minimum_spanning_tree(G)
    #for e in G.edges():
        #i, j = e
        #if i > j: i, j = j,i
        #mst_test.append( (i, j, D[i, j]) )
    #mst_test.sort(key=lambda e: e[2])

    #for i in range(len(mst)):
        #if mst[i][0] != mst_test[i][0] or mst[i][1] != mst_test[i][1]:
            #assert mst[i][2] == mst_test[i][2]
            ##print(i, mst[i], mst_test[i])

    #if not sum([m[2] for m in mst]) == sum([m[2] for m in mst_test]):
        #return False

    #if not len(np.unique([m[0] for m in mst]+[m[1] for m in mst])) == len(mst)+1:
        #return False

    #G = nx.Graph()
    #for i,j,_ in mst: G.add_edge(i, j)
    #if not nx.is_connected(G):
        #return False

    #return True
