from vqvae import VQVAE
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from dataset import NoOverlap_Dataset_EKG

model_name = '3p_str8_emb64'
ckpt = f'./checkpoint/vqvae_{model_name}_001.pt'

device = 'cuda'
vqvae_model = VQVAE()
vqvae_model.load_state_dict(torch.load(ckpt))
vqvae_model = vqvae_model.to(device)
vqvae_model.eval()

test_idx = 7500
input_len = 2048

dataset = NoOverlap_Dataset_EKG(
    root_path='/home/bryanswkim/mamba/ekg-vqvae/data/',
    data_path='ekg_SL001(SZ002, 003, 004, 005, 006)_deidentified.csv',
    size=[input_len, input_len],
)

os.makedirs(f'test_recon/{model_name}/', exist_ok=True)

# get ekg data, perform inference
sample, _ = dataset[test_idx]
sample = torch.from_numpy(sample)
sample = sample.unsqueeze(0).to(device)

output = [sample]
with torch.no_grad():
    out, _ = vqvae_model(sample)
    output.append(out)

    _, _, idx = vqvae_model.encode(sample.permute(0, 2, 1))
    print('indices: ', idx)
    print('# of indices: ', len(idx[0]))

output = torch.cat(output, 1)
output = output.detach().cpu().numpy()
output = np.squeeze(output)

sample = output[:input_len]
recons = output[input_len:]

ar = np.arange(input_len)
plt.figure(figsize=(12, 4))
plt.plot(sample, linestyle = 'solid',label = "Input")
plt.plot(recons, linestyle = 'solid',label = "Recon")
plt.legend(loc='lower right')

os.makedirs(f'test_recon/{model_name}/', exist_ok=True)
plt.savefig(f'test_recon/{model_name}/sample{input_len}_{test_idx}.png')
plt.close()