import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if 'cuda' in str(device):
    print(f'GPU number: {torch.cuda.device_count()}')
else:
    print('Using CPU.')
