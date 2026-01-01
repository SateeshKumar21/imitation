import h5py
import torch
import torch.nn as nn
import cv2
import numpy as np
from imitation.utils.lr_scheduler import get_scheduler 
from imitation.models.image_nets import AutoEncoders, VisionTransformer, ResNet18, EfficientNet, R3M
from accelerate import Accelerator
from imitation.utils.torch_utils import replace_submodules
from tqdm import tqdm

# accelerator = Accelerator()
# device = accelerator.device

if __name__ == '__main__':
    
    train, val = [], []
    with h5py.File('../data/color_lift_n200_224x224.h5', 'r') as f:
        for i in range(100):
            if i < 95:
                train.append(f['data'][f'demo_{i}']['obs']['agentview_image'][:])
            else:
                val.append(f['data'][f'demo_{i}']['obs']['agentview_image'][:])

    train = torch.tensor(np.concatenate(train), dtype=torch.float32, requires_grad=False).permute(0,3,1,2)/255
    val = torch.tensor(np.concatenate(val), dtype=torch.float32, requires_grad=False).permute(0,3,1,2)/255

    print('train shape:', train.shape)
    print('val shape:', val.shape)
    img_shape = train.shape[1:]

    autoencoder = AutoEncoders(img_shape,
                            #    encoder=ResNet18(),).to('cuda')
                               encoder=R3M(input_channel=3, r3m_model_class='resnet18', freeze=False)).to('cuda')
                            #    encoder=EfficientNet())
                            #    encoder=VisionTransformer('vit_base_patch16_clip_224.openai', pretrained=False, frozen=False))
    # autoencoder = replace_submodules(
    #             root_module=autoencoder,
    #             predicate=lambda x: isinstance(x, nn.BatchNorm2d),
    #             func=lambda x: nn.GroupNorm(
    #                 num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8), 
    #                 num_channels=x.num_features)
    #         )

    dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train.to('cuda')), batch_size=32, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(val.to('cuda')), batch_size=len(val), shuffle=False)
    
    if isinstance(autoencoder.encoder, VisionTransformer):
        optim_encoder = torch.optim.AdamW(autoencoder.encoder.parameters(), lr=3e-4, weight_decay=1e-6)
        encoder_lr_schedular = get_scheduler('cosine', optim_encoder, num_warmup_steps=2000, num_training_steps=len(dataloader)*1000)
    else:
        optim_encoder = torch.optim.Adam(autoencoder.encoder.parameters(), lr=1e-3)
        encoder_lr_schedular = torch.optim.lr_scheduler.MultiStepLR(optim_encoder, milestones=[10000], gamma=0.1)
    
    optim_decoder = torch.optim.Adam(autoencoder.decoder.parameters(), lr=1e-3)
    decoder_lr_schedular = torch.optim.lr_scheduler.MultiStepLR(optim_decoder, milestones=[10000], gamma=0.1)
    
    # autoencoder, optim_encoder, optim_decoder, encoder_lr_schedular, decoder_lr_schedular, dataloader, val_dataloader = accelerator.prepare(autoencoder, 
    #                                                                                                                         optim_encoder, 
    #                                                                                                                         optim_decoder, 
    #                                                                                                                         encoder_lr_schedular,
    #                                                                                                                         decoder_lr_schedular,
    #                                                                                                                         dataloader,
    #                                                                                                                         val_dataloader)
    loss_fn = nn.MSELoss()
    steps = 0
    for i in range(1000):
        for batch in tqdm(dataloader):
            steps += 1
            optim_encoder.zero_grad()
            optim_decoder.zero_grad()

            out = autoencoder(batch[0])
            loss = loss_fn(out, batch[0])
            loss.backward()
            # accelerator.backward(loss)
            optim_encoder.step()
            optim_decoder.step()
            encoder_lr_schedular.step()
            decoder_lr_schedular.step()
            
            # print('steps:', steps, 'train loss:', loss.item())
            if steps % 1000 == 0:
                
                with torch.no_grad():
                    val = next(iter(val_dataloader))[0]
                    autoencoder.eval()
                    out = autoencoder(val)
                    print('val loss:', loss_fn(out, val).item())
                    out = out.cpu().detach().numpy()
            
                for i in range(len(out)):
                    cv2.imshow('b', np.concatenate([val[i].cpu().detach().numpy().transpose(1,2,0), out[i].transpose(1,2,0)], axis=1))
                    cv2.waitKey(10)
                autoencoder.train()