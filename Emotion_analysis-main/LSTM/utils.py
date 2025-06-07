import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    """
    自定义的文本数据集类。
    这个类接收预处理好的数据（(ID序列, 标签)元组的列表），
    并将其转换为 PyTorch DataLoader 可以使用的格式。
    """
    def __init__(self, data):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.x = torch.LongTensor([x[0] for x in data]).to(self.device)
        self.y = torch.LongTensor([x[1] for x in data]).to(self.device)
    def __getitem__(self,index):
        self.text = self.x[index]
        self.label = self.y[index]
        return self.text, self.label
    def __len__(self):
        return len(self.x)