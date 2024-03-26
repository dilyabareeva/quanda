from explainers import Explainer
from metrics import Metric
from torchvision.datasets import MNIST
from torchvision.models import resnet18
from torch.utils.data import DataLoader
train_ds=MNIST(root="~/Documents/Code/Datasets",train=True)
test_ds=MNIST(root="~/Documents/Code/Datasets",train=False)
test_ld=DataLoader(test_ds,batch_size=32)
model=resnet18()
# Possibly get special kinds of datasets here
metric=Metric(train_ds,test_ds)
# Possibly train model on the special kind of dataset with something like metric.train_model()
explainer=Explainer(model,train_ds,"cuda")
explainer.train()
for x,y in iter(test_ld):
    preds=model(x).argmax(dim=-1)
    xpl=explainer.explain(x,preds)
    metric(xpl)
metric.get_result()