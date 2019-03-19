import time, sys, datetime
import numpy as np
from evaluate import evaluation_4class

import TD_RvNN

obj = "Twitter15" # choose dataset, you can choose either "Twitter15" or "Twitter16"
fold = "2" # fold index, choose from 0-4
tag = "_u2b"
vocabulary_size = 5000
hidden_dim = 100
Nclass = 4
Nepoch = 600
lr = 0.005

unit="TD_RvNN-"+obj+str(fold)+'-vol.'+str(vocabulary_size)+tag
#lossPath = "../loss/loss-"+unit+".txt"
#modelPath = "../param/param-"+unit+".npz"

treePath = './resource/data.TD_RvNN.vol_'+str(vocabulary_size)+'.txt'

trainPath = "./nfold/RNNtrainSet_"+obj+str(fold)+"_tree.txt"
testPath = "./nfold/RNNtestSet_"+obj+str(fold)+"_tree.txt"
labelPath = "./resource/"+obj+"_label_All.txt"

#### tools ####
from tools import loadLabel, constructTree

### load data ####
def loadData():
	print("loading tree label")
	labelDic = {}
	for line in open(labelPath):
		line = line.rstrip()
		label, eid = line.split('\t')[0], line.split('\t')[2]
		labelDic[eid] = label.lower()
	print(len(labelDic))

	print("reading tree")  ## X
	treeDic = {}
	for line in open(treePath):
		line = line.rstrip()
		eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
		parent_num, maxL = int(line.split('\t')[3]), int(line.split('\t')[4])
		Vec = line.split('\t')[5]
		if eid not in treeDic:
		# if not treeDic.has_key(eid):
			treeDic[eid] = {}
		treeDic[eid][indexC] = {'parent': indexP, 'parent_num': parent_num, 'maxL': maxL, 'vec': Vec}
	print('tree no:', len(treeDic))

	print("loading train set")
	tree_train, word_train, index_train, y_train, parent_num_train, c = [], [], [], [], [], 0
	l1, l2, l3, l4 = 0, 0, 0, 0
	for eid in open(trainPath):
		# if c > 8: break
		eid = eid.rstrip()
		if eid not in labelDic: continue
		if eid not in treeDic: continue
		# if not labelDic.has_key(eid): continue
		# if not treeDic.has_key(eid): continue
		if len(treeDic[eid]) <= 0:
			# print labelDic[eid]
			continue
		## 1. load label
		label = labelDic[eid]
		y, l1, l2, l3, l4 = loadLabel(label, l1, l2, l3, l4)
		y_train.append(y)
		## 2. construct tree
		# print eid
		x_word, x_index, tree, parent_num = constructTree(treeDic[eid])
		tree_train.append(tree)
		word_train.append(x_word)
		index_train.append(x_index)
		parent_num_train.append(parent_num)
		# print treeDic[eid]
		# print tree, child_num
		# exit(0)
		c += 1
	print(l1, l2, l3, l4)

	print("loading test set")
	tree_test, word_test, index_test, parent_num_test, y_test, c = [], [], [], [], [], 0
	l1, l2, l3, l4 = 0, 0, 0, 0
	for eid in open(testPath):
		# if c > 4: break
		eid = eid.rstrip()
		if eid not in labelDic: continue
		if eid not in treeDic: continue
		# if not labelDic.has_key(eid): continue
		# if not treeDic.has_key(eid): continue
		if len(treeDic[eid]) <= 0:
			# print labelDic[eid]
			continue
		## 1. load label
		label = labelDic[eid]
		y, l1, l2, l3, l4 = loadLabel(label, l1, l2, l3, l4)
		y_test.append(y)
		## 2. construct tree
		x_word, x_index, tree, parent_num = constructTree(treeDic[eid])
		tree_test.append(tree)
		word_test.append(x_word)
		index_test.append(x_index)
		parent_num_test.append(parent_num)
		c += 1
	print(l1, l2, l3, l4)
	print("train no:", len(tree_train), len(word_train), len(index_train), len(parent_num_train), len(y_train))
	print("test no:", len(tree_test), len(word_test), len(index_test), len(parent_num_test), len(y_test))
	print("dim1 for 0:", len(tree_train[0]), len(word_train[0]), len(index_train[0]))
	print("case 0:", tree_train[0][0], word_train[0][0], index_train[0][0], parent_num_train[0])
	# print index_train[0]
	# print word_train[0]
	# print tree_train[0]
	# exit(0)
	return tree_train, word_train, index_train, parent_num_train, y_train, tree_test, word_test, index_test, parent_num_test, y_test

### main ###
tree_train, word_train, index_train, parent_num_train, y_train, tree_test, word_test, index_test, parent_num_test, y_test = loadData()
t0 = time.time()
model = TD_RvNN.RvNN(vocabulary_size, hidden_dim, Nclass)
t1 = time.time()
print('Recursive model established,', (t1 - t0) / 60)

losses_5, losses = [], []
num_examples_seen = 0
for epoch in range(Nepoch):
	## one SGD
	indexs = [i for i in range(len(y_train))]
	# random.shuffle(indexs)
	for i in indexs:
		'''print i,":", len(tree_train[i])
		print tree_train[i]
		tree_state = model._state(word_train[i], index_train[i], child_num_train[i], tree_train[i])
		print len(tree_state)
		print tree_state
		evl = model._evaluate(word_train[i], index_train[i], child_num_train[i], tree_train[i])
		print len(evl) 
		print evl'''
		loss, pred_y = model.train_step_up(word_train[i], index_train[i], parent_num_train[i], tree_train[i],
		                                   y_train[i], lr)
		# print loss, pred_y
		losses.append(loss.round(2))
		# losses.append(round(loss, 2))
		'''if math.isnan(loss):
		#   continue 
		   print loss, pred_y
		   print i
		   print len(tree_train[i]), len(word_train[i]), parent_num_train[i]
		   print tree_train[i]
		   print word_train[i]
		   print 'final_state:',model._evaluate(word_train[i], index_train[i], parent_num_train[i], tree_train[i])'''
		num_examples_seen += 1
	print("epoch=%d: loss=%f" % (epoch, np.mean(losses)))
	sys.stdout.flush()

	## cal loss & evaluate
	if epoch % 5 == 0:
		losses_5.append((num_examples_seen, np.mean(losses)))
		time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, np.mean(losses)))
		# floss.write(str(time)+": epoch="+str(epoch)+" loss="+str(loss) +'\n')
		# floss.flush()
		sys.stdout.flush()
		prediction = []
		for j in range(len(y_test)):
			# print j
			prediction.append(model.predict_up(word_test[j], index_test[j], parent_num_test[j], tree_test[j]))
		res = evaluation_4class(prediction, y_test)
		print('results:', res)
		sys.stdout.flush()
		## Adjust the learning rate if loss increases
		if len(losses_5) > 1 and losses_5[-1][1] > losses_5[-2][1]:
			lr = lr * 0.5
			print("Setting learning rate to %f" % lr)
			sys.stdout.flush()

	losses = []
