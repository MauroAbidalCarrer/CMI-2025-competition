import hiddenlayer as hl
from torch.utils.data import DataLoader as DL

from model import mk_model
from dataset import split_dataset


train_dataset, seq_meta = split_dataset()["train"]
train_reg_demos_y = train_dataset.tensors[4]
model = mk_model(
    dataset_x=train_dataset.tensors[0],
    reg_demos_dataset_y=train_reg_demos_y,
)
data_loader = DL(train_dataset)
first_batch = next(iter(data_loader))
# Build a graph
graph = hl.build_graph(model, first_batch)
graph.save("model_structure", format="png")
