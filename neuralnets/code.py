########################
# Problem 2
########################
import numpy as np
import matplotlib.pyplot as plt

inputs = np.random.normal(0.0, 1.0, (256, 1000))
fcs = np.random.normal(0.0, np.sqrt(2.0/256.0), (256, 256, 10))

def forward(x):
    results = []
    for i in range(10):
        x = np.matmul(fcs[:, :, i], x)
        x = np.maximum(x, 0)
        results.append(x)

    results = np.asarray(results)
    return results

def compute_params(results):
        means = np.mean(results, axis=1)
        stds = np.std(results, axis=1)
        return means, stds

all_means = []
all_stds = []
all_results = []
for i in range(1000):
    results = forward(inputs[:, i])
#     print(np.shape(results))
    means, stds = compute_params(results)
#     print(np.shape(means))
    all_means.append(means)
    all_stds.append(stds)
    all_results.append(results)
    
all_means = np.asarray(all_means)
print(np.shape(all_means))
all_stds = np.asarray(all_stds)
all_results = np.asarray(all_results)

layer_mean = np.mean(all_means, axis=0)
layer_std = np.std(all_stds, axis=0)
print(np.shape(layer_mean))

plt.plot(np.arange(1, 11), layer_mean, np.arange(1, 11), layer_std)
plt.title('Parameters Per Layer')
plt.xlabel('Layer')
plt.legend(['$\mu$', '$\sigma$'])
plt.show()

layer2 = all_results[:, 1, :]
layer4 = all_results[:, 3, :]
layer6 = all_results[:, 5, :]
layer8 = all_results[:, 7, :]
layer10 = all_results[:, 9, :]

layer2 = np.reshape(layer2, (1000 * 256, 1))
layer4 = np.reshape(layer4, (1000 * 256, 1))
layer6 = np.reshape(layer6, (1000 * 256, 1))
layer8 = np.reshape(layer8, (1000 * 256, 1))
layer10 = np.reshape(layer10, (1000 * 256, 1))

plt.hist(layer2)
plt.title('Layer 2 Output Histogram')
plt.show()
plt.hist(layer4)
plt.title('Layer 4 Output Histogram')
plt.show()
plt.hist(layer6)
plt.title('Layer 6 Output Histogram')
plt.show()
plt.hist(layer8)
plt.title('Layer 8 Output Histogram')
plt.show()
plt.hist(layer10)
plt.title('Layer 10 Output Histogram')
plt.show()


########################
# Problem 3
########################
import torch
import numpy as np
import matplotlib.pyplot as plt

mean1 = torch.load('mean1.pt')
mean2 = torch.load('mean2.pt')
std1 = torch.load('var1.pt')
std2 = torch.load('var2.pt')
layer1 = torch.load('layer1.pt')
layer3 = torch.load('layer3.pt')
layer5 = torch.load('layer5.pt')
layer7 = torch.load('layer7.pt')
layer3_1 = torch.load('layer3_1.pt')
layer5_1 = torch.load('layer5_1.pt')
layer7_1 = torch.load('layer7_1.pt')

layer1_np = layer1.detach().numpy()
layer3_np = layer3.detach().numpy()
layer5_np = layer5.detach().numpy()
layer7_np = layer7.detach().numpy()
layer3_1_np = layer3.detach().numpy()
layer5_1_np = layer5.detach().numpy()
layer7_1_np = layer7.detach().numpy()

for i in range(len(mean1)):
    if i > 0:
        mean1[i] = mean1[i].detach().numpy()
        std1[i] = std1[i].detach().numpy()
        mean2[i] = mean2[i].detach().numpy()
        std2[i] = std2[i].detach().numpy()

size1 = np.shape(layer1_np)
size3 = np.shape(layer3_np)
size5 = np.shape(layer5_np)
size7 = np.shape(layer7_np)

layer1_np = np.reshape(layer1_np, (size1[0]*size1[1]*size1[2]*size1[3], 1))
layer3_np = np.reshape(layer3_np, (size3[0]*size3[1]*size3[2]*size3[3], 1))
layer5_np = np.reshape(layer5_np, (size5[0]*size5[1]*size5[2]*size5[3], 1))
layer7_np = np.reshape(layer7_np, (size7[0]*size7[1]*size7[2]*size7[3], 1))

plt.hist(layer1_np)
plt.title('Layer 1 Output')
plt.show()

plt.hist(layer3_np)
plt.title('Layer 3 Output')
plt.show()

plt.hist(layer5_np)
plt.title('Layer 5 Output')
plt.show()

plt.hist(layer7_np)
plt.title('Lay

mean2_1 = np.repeat(mean2[0], size1[0]*size1[2]*size1[3])
mean2_3 = np.repeat(mean2[1], size3[0]*size3[2]*size3[3])
mean2_5 = np.repeat(mean2[2], size5[0]*size5[2]*size5[3])
mean2_7 = np.repeat(mean2[3], size7[0]*size7[2]*size7[3])

var2_1 = np.repeat(std2[0], size1[0]*size1[2]*size1[3])
var2_3 = np.repeat(std2[1], size3[0]*size3[2]*size3[3])
var2_5 = np.repeat(std2[2], size5[0]*size5[2]*size5[3])
var2_7 = np.repeat(std2[3], size7[0]*size7[2]*size7[3])

layer1_np_bn = layer1_np[:, 0] - mean2_1.numpy()
layer3_np_bn = layer3_np[:, 0] - mean2_3
layer5_np_bn = layer5_np[:, 0] - mean2_5
layer7_np_bn = layer7_np[:, 0] - mean2_7

layer1_np_bn = np.divide(layer1_np_bn, np.sqrt(var2_1))
layer3_np_bn = np.divide(layer3_np_bn, np.sqrt(var2_3))
layer5_np_bn = np.divide(layer5_np_bn, np.sqrt(var2_5))
layer7_np_bn = np.divide(layer7_np_bn, np.sqrt(var2_7))

plt.hist(layer1_np_bn)
plt.title('Layer 1 Output Normalized')
plt.show()

plt.hist(layer3_np_bn)
plt.title('Layer 3 Output Normalized')
plt.show()

plt.hist(layer5_np_bn)
plt.title('Layer 5 Output Normalized')
plt.show()

plt.hist(layer7_np_bn)
plt.title('Layer 7 Output Normalized')
plt.show()

########################
# Problem 4 Snippets
########################


        # 1st 5 layers
#     for params in model.module.conv1.parameters():
#         params.requires_grad = False
    
#     for params in model.module.bn1.parameters():
#         params.requires_grad = False
        
#     for params in model.module.layer1.parameters():
#         params.requires_grad = False
    
# #     # 1st 9 layers
#     for params in model.module.layer2.parameters():
#         params.requires_grad = False
        
# #     # 1st 13 layers
#     for params in model.module.layer3.parameters():
#         params.requires_grad = False
        
# #     # 1st 17 layers
#     for params in model.module.layer4.parameters():
#         params.requires_grad = False

def batch_test(val_loader, model):
    batch_time = AverageMeter()
    
    model.eval()
    
    end = time.time()
    mean1 = []
    std1 = []
    mean2 = []
    std2 = []
    for i, (input, target) in enumerate(val_loader):
#         input_batch = torch.autograd.Variable(input, volatile=True)
        input_sample = torch.split(input, 10)
        output = model(input_sample[0])
        
        layer1 = model.module.pre_bn1.cpu()
        
        layer3_1 = model.module.layer1[0].pre_bn1.cpu()
        layer5_1 = model.module.layer2[0].pre_bn1.cpu()
        layer7_1 = model.module.layer3[0].pre_bn1.cpu()
        layer3 = model.module.layer1[0].pre_bn2.cpu()
        layer5 = model.module.layer2[0].pre_bn2.cpu()
        layer7 = model.module.layer3[0].pre_bn2.cpu()

        mean1.append(model.module.bn1.running_mean.cpu())
        mean1.append(model.module.layer1[0].bn1.running_mean.cpu())
        mean1.append(model.module.layer2[0].bn1.running_mean.cpu())
        mean1.append(model.module.layer3[0].bn1.running_mean.cpu())
        
        std1.append(model.module.bn1.running_var.cpu())
        std1.append(model.module.layer1[0].bn1.running_var.cpu())
        std1.append(model.module.layer2[0].bn1.running_var.cpu())
        std1.append(model.module.layer3[0].bn1.running_var.cpu())

        mean2.append(model.module.bn1.running_mean.cpu())
        mean2.append(model.module.layer1[0].bn2.running_mean.cpu())
        mean2.append(model.module.layer2[0].bn2.running_mean.cpu())
        mean2.append(model.module.layer3[0].bn2.running_mean.cpu())
        
        std2.append(model.module.bn1.running_var.cpu())
        std2.append(model.module.layer1[0].bn2.running_var.cpu())
        std2.append(model.module.layer2[0].bn2.running_var.cpu())
        std2.append(model.module.layer3[0].bn2.running_var.cpu())

        torch.save(layer3_1, 'layer3_1.pt')
        torch.save(layer5_1, 'layer5_1.pt')
        torch.save(layer7_1, 'layer7_1.pt')
        torch.save(layer1, 'layer1.pt')
        torch.save(layer3, 'layer3.pt')
        torch.save(layer5, 'layer5.pt')
        torch.save(layer7, 'layer7.pt')
        torch.save(mean1, 'mean1.pt')
        torch.save(mean2, 'mean2.pt')
        torch.save(std1, 'var1.pt')
        torch.save(std2, 'var2.pt')
        break