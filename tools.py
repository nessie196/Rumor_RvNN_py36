import TD_RvNN

def str2matrix(Str, MaxL): # str = index:wordfreq index:wordfreq
    wordFreq, wordIndex = [], []
    l = 0
    for pair in Str.split(' '):
        wordFreq.append(float(pair.split(':')[1]))
        wordIndex.append(int(pair.split(':')[0]))
        l += 1
    ladd = [ 0 for i in range( MaxL-l ) ]
    wordFreq += ladd
    wordIndex += ladd
    #print MaxL, l, len(Str.split(' ')), len(wordFreq)
    #print Str.split(' ')
    return wordFreq, wordIndex

def loadLabel(label, l1, l2, l3, l4):
    labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'non-rumor'], ['false'], ['true'], ['unverified']
    if label in labelset_nonR:
       y_train = [1,0,0,0]
       l1 += 1
    if label in labelset_f:
       y_train = [0,1,0,0]
       l2 += 1
    if label in labelset_t:
       y_train = [0,0,1,0]
       l3 += 1
    if label in labelset_u:
       y_train = [0,0,0,1]
       l4 += 1
    return y_train, l1,l2,l3,l4

def constructTree(tree):
    ## tree: {index1:{'parent':, 'maxL':, 'vec':}
    ## 1. ini tree node
    index2node = {}
    for i in tree:
        node = TD_RvNN.Node_tweet(idx=i)
        index2node[i] = node
    ## 2. construct tree
    for j in tree:
        indexC = j
        indexP = tree[j]['parent']
        nodeC = index2node[indexC]
        wordFreq, wordIndex = str2matrix( tree[j]['vec'], tree[j]['maxL'] )
        #print tree[j]['maxL']
        nodeC.index = wordIndex
        nodeC.word = wordFreq
        #nodeC.time = tree[j]['post_t']
        ## not root node ##
        if not indexP == 'None':
           nodeP = index2node[int(indexP)]
           nodeC.parent = nodeP
           nodeP.children.append(nodeC)
        ## root node ##
        else:
           root = nodeC
    ## 3. convert tree to DNN input
    parent_num = tree[j]['parent_num']
    ini_x, ini_index = str2matrix( "0:0", tree[j]['maxL'] )
    #x_word, x_index, tree = tree_gru_u2b.gen_nn_inputs(root, ini_x, ini_index)
    x_word, x_index, tree = TD_RvNN.gen_nn_inputs(root, ini_x)
    return x_word, x_index, tree, parent_num