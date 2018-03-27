#/usr/bin/env python
# --*-- coding: utf-8 --*--
import os
import gzip
import numpy as np
import cPickle
import matplotlib.pyplot as plt

# caffe2 packages
from caffe2.python import core, workspace, model_helper, brew, optimizer
from caffe2.proto import caffe2_pb2

##########################################
#  åload Fashion Mnist ædataset®
##########################################
def load_fashion_mnist(path, kind='train'):
    """Load Fashion MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

# åload dataset®
raw_X_train, raw_y_train = load_fashion_mnist('data/', kind='train') # 60000 for training¬
X_test, y_test = load_fashion_mnist('data/', kind='t10k') # 10000 for test¬

# ætrain:val ==> 8:2
X_train = raw_X_train[:int(0.8 * raw_X_train.shape[0])]
X_val = raw_X_train[int(0.8 * raw_X_train.shape[0]):]
y_train = raw_y_train[:int(0.8 * raw_X_train.shape[0])]
y_val = raw_y_train[int(0.8 * raw_X_train.shape[0]):]
print X_train.shape, X_val.shape
# (48000, 784) (12000, 784)


##########################################
#  å®Define LeNet network
##########################################
# æEnable with NV GPU device
device_option = caffe2_pb2.DeviceOption(device_type=caffe2_pb2.CUDA)

def AddLeNetModel(model):
    with core.DeviceScope(device_option):
        conv1 = brew.conv(model,'data', 'conv1', 1, 20, 5)
        pool1 = brew.max_pool(model, conv1, 'pool1', kernel=2, stride=2)
        conv2 = brew.conv(model, pool1, 'conv2', 20, 50, 5)
        pool2 = brew.max_pool(model, conv2, 'pool2', kernel=2, stride=2)
        fc3 = brew.fc(model, pool2, 'fc3', 50 * 4 * 4, 500)
        fc3 = brew.relu(model, fc3, fc3)
        pred = brew.fc(model, fc3, 'pred', 500, 10)
        softmax = brew.softmax(model, pred, 'softmax')
    return softmax
# Accuracy
def AddAccuracy(model, softmax):
    accuracy = brew.accuracy(model, [softmax, 'label'], "accuracy")
    return accuracy


##########################################
#  å®Define training Ops
##########################################
# ècalculate cross entropy loss
# écalculate the accuracy on the training setç²¾åº¦
def AddTrainingOperators(model, softmax):
    # calculate Loss
    xent = model.LabelCrossEntropy([softmax, 'label'])
    loss = model.AveragedLoss(xent, "loss")
    # calculate Accuracy
    AddAccuracy(model, softmax)
    # Add loss to gradient for backpropogation
    model.AddGradientOperators([loss])
    # Init SGD optimizer solver
    opt = optimizer.build_sgd(model, base_learning_rate=0.1, policy="step", stepsize=1, gamma=0.999)


##########################################
#  åDefine Training mode
##########################################
batch_size = 64
# reset workspace
workspace.ResetWorkspace()
training_model = model_helper.ModelHelper(name="training_net")
gpu_id = 0 # Use gpu ID 0
training_model.net.RunAllOnGPU(gpu_id=gpu_id, use_cudnn=True)
training_model.param_init_net.RunAllOnGPU(gpu_id=gpu_id, use_cudnn=True)
# Add LeNet network and Ops
soft=AddLeNetModel(training_model)
AddTrainingOperators(training_model, soft)
# Init training mode for workspace
workspace.RunNetOnce(training_model.param_init_net)
workspace.CreateNet(training_model.net,overwrite=True,input_blobs=['data','label'])


##########################################
#  äSave snapshots
##########################################
# ä¿Save training network weight; as {key-blob name; value-weights}
snapshot_location = 'snapshots/'
def save_snapshot(model,iter_num):
    snapshot = {}
    for blob in model.GetParams():
        snapshot[blob]=workspace.FetchBlob(blob)
    cPickle.dump(snapshot, open(snapshot_location + str(iter_num),'w'))


##########################################
#  åCreat Validation  Model
##########################################
val_model = model_helper.ModelHelper(name="validation_net", init_params=False)
val_model.net.RunAllOnGPU(gpu_id=gpu_id, use_cudnn=True)
val_model.param_init_net.RunAllOnGPU(gpu_id=gpu_id, use_cudnn=True)
val_soft = AddLeNetModel(val_model)
AddAccuracy(val_model,val_soft)
workspace.RunNetOnce(val_model.param_init_net)
workspace.CreateNet(val_model.net,overwrite=True,input_blobs=['data','label'])


##########################################
#  åEstimate model performance in validation set°
##########################################
# è¿Return average loss and accuracy
def check_val():
    accuracy = []
    start=0
    while start < X_val.shape[0]:
        label = y_val[start : start + batch_size].astype(np.int32)
        batch = X_val[start : start + batch_size, :].reshape(label.shape[0], 28, 28)
        batch = batch[:, np.newaxis,...].astype(np.float32)
        batch = batch * float(1./256)
        workspace.FeedBlob("data", batch, device_option)
        workspace.FeedBlob("label", label, device_option)
        workspace.RunNet(val_model.net, num_iter=1)
        accuracy.append(workspace.FetchBlob('accuracy'))
        start += label.shape[0]
    return np.mean(accuracy)


# Set the number of iterations required to output snapshots°
total_iterations = 710
snapshot_interval = 100
total_iterations = total_iterations * 64
print workspace.Blobs()
# [u'ONE_1_0', u'SgdOptimizer_0_lr_gpu0', u'accuracy', u'conv1', u'conv1_b', u'conv1_b_grad', u'conv1_grad', u'conv1_w', u'conv1_w_grad', u'conv2', u'conv2_b', u'conv2_b_grad', u'conv2_grad', u'conv2_w', u'conv2_w_grad', u'data', u'data_grad', u'fc3', u'fc3_b', u'fc3_b_grad', u'fc3_grad', u'fc3_w', u'fc3_w_grad', u'iteration_mutex', u'label', u'loss', u'loss_autogen_grad', u'optimizer_iteration', u'pool1', u'pool1_grad', u'pool2', u'pool2_grad', u'pred', u'pred_b', u'pred_b_grad', u'pred_grad', u'pred_w', u'pred_w_grad', u'softmax', u'softmax_grad', u'training_net_1/LabelCrossEntropy', u'training_net_1/LabelCrossEntropy_grad']


##########################################
#  å¼Start training
##########################################
accuracy = []
val_accuracy = []
loss = []
lr = []
start=0
while start < total_iterations:
    label = y_train[start : start + batch_size].astype(np.int32)
    data = X_train[start : start + batch_size, :].reshape(label.shape[0], 28, 28)
    data = data[:,np.newaxis,...].astype(np.float32)
    data = data * float(1./256) # Scaling the pixel values for faster computation
    workspace.FeedBlob("data", data, device_option)
    workspace.FeedBlob("label", label, device_option)
    workspace.RunNet(training_model.net, num_iter=1)
    accuracy.append(workspace.FetchBlob('accuracy'))
    loss.append(workspace.FetchBlob('loss'))
    lr.append(workspace.FetchBlob('SgdOptimizer_0_lr_gpu0')) # learning rate å€¼
    # lr.append(workspace.FetchBlob('conv1_b_lr'))
    if start%snapshot_interval == 0:
        save_snapshot(training_model,start)
        #print 'iterations:', loss, '\n'	 
    val_accuracy.append(check_val())
    start += batch_size


##########################################
#  åVisualization of training result 
##########################################
#plt.subplot(211)
#plt.plot(accuracy, 'b', label='Training Set')
#plt.plot(val_accuracy, 'r', label='Validation Set')
#plt.ylabel('Accuracy')
#plt.xlabel('Num of Iterations')
#plt.legend(loc=4)

#plt.subplot(212)
#plt.plot(loss, 'b', label='Training Set')
#plt.ylabel('Loss')
#plt.xlabel('Num of Iterations')
#plt.legend(loc=1)
#plt.show()


##########################################
#  åCreat Test Model
##########################################
test_model = model_helper.ModelHelper(name="testing_net", init_params=False)
test_model.net.RunAllOnGPU(gpu_id=gpu_id, use_cudnn=True)
test_model.param_init_net.RunAllOnGPU(gpu_id=gpu_id, use_cudnn=True)
test_soft=AddLeNetModel(test_model)
workspace.RunNetOnce(test_model.param_init_net)
workspace.CreateNet(test_model.net,overwrite=True,input_blobs=['data'])

# æFinding the best training model snapshot on validation set
best = np.argmax(np.array(val_accuracy)[range(0,np.array(val_accuracy).shape[0], snapshot_interval)])
best = best * batch_size * snapshot_interval
# åLoad the best model into workspace
params=cPickle.load(open(snapshot_location + str(best),'rb'))
for blob in params.keys():
    workspace.FeedBlob(blob, params[blob], device_option)


##########################################
#  åPredict output on the test data setº
##########################################
results = []
start = 0
count = 0
while start < X_test.shape[0]:
    raw_batch = X_test[start : start + batch_size, :]
    labels = y_test[start : start + batch_size]
    batch = raw_batch.reshape(raw_batch.shape[0],28,28)
    batch = batch[:, np.newaxis,...].astype(np.float32)
    batch = batch * float(1./256)
    workspace.FeedBlob("data", batch, device_option)
    workspace.RunNet(test_model.net, num_iter=1)
    res = np.argmax(workspace.FetchBlob('softmax'), axis=1)
    feat = workspace.FetchBlob('fc3')

    for i in range(raw_batch.shape[0]):
        if res[i] == labels[i]:
            count += 1
    start += raw_batch.shape[0]

print 'Top-1 Testing Accuracy: ', count/float(X_test.shape[0])
# Top-1 Testing Accuracy:  0.7774

print 'Done.'
