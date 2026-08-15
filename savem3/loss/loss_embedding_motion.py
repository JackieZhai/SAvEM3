import torch
import torch.nn as nn
import torch.nn.functional as F


def smooth_loss(embedding, interp=2, cosine=False, delta=0.5, weightmap=None):
    assert weightmap is None
    embedding = F.interpolate(embedding, scale_factor=(1, 1/interp, 1/interp), mode='trilinear', align_corners=True)

    batch_size = embedding.shape[0]
    loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)

    if cosine:
        loss += torch.sum(1 - F.cosine_similarity(embedding[:, :, :, :, 1:], embedding[:, :, :, :, :-1], dim=1))
        loss += torch.sum(1 - F.cosine_similarity(embedding[:, :, :, 1:, :], embedding[:, :, :, :-1, :], dim=1))
        loss += torch.sum(1 - F.cosine_similarity(embedding[:, :, 1:, :, :], embedding[:, :, :-1, :, :], dim=1))
    else:
        loss += torch.sum(F.relu(torch.norm(embedding[:, :, :, :, 1:] - embedding[:, :, :, :, :-1]) - delta)**2)
        loss += torch.sum(F.relu(torch.norm(embedding[:, :, :, 1:, :] - embedding[:, :, :, :-1, :]) - delta)**2)
        loss += torch.sum(F.relu(torch.norm(embedding[:, :, 1:, :, :] - embedding[:, :, :-1, :, :]) - delta)**2)

    loss = loss / batch_size
    return loss


def smooth_weighted_loss(embedding, interp=2, cosine=False, delta=0.5, weightmap=None):
    embedding = F.interpolate(embedding, scale_factor=(1, 1/interp, 1/interp), mode='trilinear', align_corners=True)

    batch_size = embedding.shape[0]
    loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)

    if cosine:
        loss += torch.sum(1 - F.cosine_similarity(embedding[:, :, :, :, 1:], embedding[:, :, :, :, :-1], dim=1))
        loss += torch.sum(1 - F.cosine_similarity(embedding[:, :, :, 1:, :], embedding[:, :, :, :-1, :], dim=1))
        loss += torch.sum(1 - F.cosine_similarity(embedding[:, :, 1:, :, :], embedding[:, :, :-1, :, :], dim=1))
    else:
        if weightmap is None:
            loss += torch.sum(F.relu(torch.norm(embedding[:, :, :, :, 1:] - embedding[:, :, :, :, :-1]) - delta)**2)
            loss += torch.sum(F.relu(torch.norm(embedding[:, :, :, 1:, :] - embedding[:, :, :, :-1, :]) - delta)**2)
            loss += torch.sum(F.relu(torch.norm(embedding[:, :, 1:, :, :] - embedding[:, :, :-1, :, :]) - delta)**2)
        else:
            loss += torch.sum(weightmap[:, :, :, :, 1:] * F.relu(torch.norm(embedding[:, :, :, :, 1:] - embedding[:, :, :, :, :-1]) - delta)**2)
            loss += torch.sum(weightmap[:, :, :, 1:, :] * F.relu(torch.norm(embedding[:, :, :, 1:, :] - embedding[:, :, :, :-1, :]) - delta)**2)
            loss += torch.sum(weightmap[:, :, 1:, :, :] * F.relu(torch.norm(embedding[:, :, 1:, :, :] - embedding[:, :, :-1, :, :]) - delta)**2)

    loss = loss / batch_size
    return loss


def motion_loss(embedding, flow, interp=2, cosine=False, delta=0.5):
    embedding = F.interpolate(embedding, scale_factor=(1, 1/interp, 1/interp), mode='trilinear', align_corners=True)
    flow = flow[:, :, :, ::interp, ::interp]

    batch_size = embedding.shape[0]
    D = embedding.shape[2]
    H = embedding.shape[3]
    W = embedding.shape[4]
    assert flow.shape[1] == 2
    assert embedding.shape[2] == flow.shape[2]
    assert embedding.shape[3] == flow.shape[3]
    loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)

    flow = flow.data.cpu()
    
    for b in range(batch_size):
        embedding_b = embedding[b] # (C, D, H, W)
        flow_b = flow[b] # (2, D, H, W)

        for z in range(D-1):
            flow_z = flow_b[:, z]
            embedding_z = embedding_b[:, z]
            embedding_z1 = embedding_b[:, z+1]
            for y in range(H):
                for x in range(W):
                    x = torch.tensor(x, dtype=torch.long)
                    y = torch.tensor(y, dtype=torch.long)
                    fx = flow_z[0, y, x]; fy = flow_z[1, y, x]
                    if torch.isnan(fx) or torch.isnan(fy):  # skip NaN flows
                        continue
                    x1 = x + fx; y1 = y + fy
                    if x1 < 0 or x1 >= W or y1 < 0 or y1 >= H:  # skip out-of-boundary locations
                        continue
                    x1 = torch.tensor(x1, dtype=torch.long)
                    y1 = torch.tensor(y1, dtype=torch.long)
                    if cosine:
                        loss += 1 - F.cosine_similarity(embedding_z1[:, y1, x1], embedding_z[:, y, x], dim=0)
                    else:
                        loss += F.relu(torch.norm(embedding_z1[:, y1, x1] - embedding_z[:, y, x]) - delta)**2

    # print(loss.data.cpu().numpy()/batch_size/D/H/W)
    loss = loss / batch_size
    return loss


def motion_loss_fast(embedding, flow, interp=2, cosine=False, delta=0.5):
    B, C, D, H, W = embedding.shape
    assert flow.shape[1] == 2
    assert embedding.shape[2] == flow.shape[2]
    assert embedding.shape[3] == flow.shape[3]
    loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
    

    for z in range(D-1):
        flow_z = flow[:, :, z, ::interp, ::interp]  # (B, 2, H/2, W/2)
        embedding_z = embedding[:, :, z, :, :]  # (B, C, H, W)
        embedding_z1 = embedding[:, :, z+1, :, :]  # (B, C, H, W)
        coord_z_y = torch.arange(H, device=embedding.device).view(1, -1).expand(B, -1).view(B, H, 1).expand(B, H, W)  # (B, H, W)
        coord_z_x = torch.arange(W, device=embedding.device).view(1, -1).expand(B, -1).view(B, 1, W).expand(B, H, W)  # (B, H, W)
        coord_z = torch.stack([coord_z_x, coord_z_y], dim=1)  # (B, 2, H, W)
        coord_z1 = coord_z[:, :, ::interp, ::interp] + flow_z  # (B, 2, H/2, W/2)
        coord_z1_norm = coord_z1 / torch.tensor([W, H], dtype=coord_z1.dtype, device=coord_z1.device).view(1, 2, 1, 1) * 2 - 1

        embedding_z1_f = F.grid_sample(embedding_z1, coord_z1_norm.permute(0, 2, 3, 1), 
                                       padding_mode='zeros', align_corners=True)  # (B, C, H/2, W/2)
        embedding_z_f = F.interpolate(embedding_z, scale_factor=(1/interp, 1/interp), 
                                      mode='bilinear', align_corners=True)  # (B, C, H/2, W/2)

        coord_mask = (coord_z1[:, 0] >= 0) & (coord_z1[:, 0] < W) & (coord_z1[:, 1] >= 0) & (coord_z1[:, 1] < H)  # (B, H, W)
        coord_mask = coord_mask.unsqueeze(1)  # (B, 1, H, W)
        if cosine:
            loss += torch.sum((1 - F.cosine_similarity(embedding_z1_f, embedding_z_f, dim=1)) * coord_mask.float())
        else:
            loss += torch.sum(F.relu(torch.abs(embedding_z1_f - embedding_z_f) * coord_mask.float() - delta) ** 2)

    # print(loss.data.cpu().numpy()/B/C/D/H/W)
    loss /= B
    return loss


def motion_loss_fast_2(embedding, flow, interp=2, cosine=True, delta=0.5):
    B, C, D, H, W = embedding.shape
    assert flow.shape[1] == 2
    assert embedding.shape[2] == flow.shape[2]
    assert embedding.shape[3] == flow.shape[3]
    
    flow_z = flow[:, :, :-1, ::interp, ::interp]  # (B, 2, D-1, H/2, W/2)
    embedding_z = embedding[:, :, :-1, :, :]  # (B, C, D-1, H, W)
    embedding_z1 = embedding[:, :, 1:, :, :]  # (B, C, D-1, H, W)

    coord_z_z = torch.arange(D-1, device=embedding.device).view(1, -1).expand(B, -1).view(B, D-1, 1, 1).expand(B, D-1, H, W)  # (B, D-1, H, W)
    coord_z_y = torch.arange(H, device=embedding.device).view(1, -1).expand(B, -1).view(B, 1, H, 1).expand(B, D-1, H, W)  # (B, D-1, H, W)
    coord_z_x = torch.arange(W, device=embedding.device).view(1, -1).expand(B, -1).view(B, 1, 1, W).expand(B, D-1, H, W)  # (B, D-1, H, W)
    coord_z = torch.stack([coord_z_x, coord_z_y, coord_z_z], dim=1)  # (B, 3, D-1, H, W)
    
    flow_z = torch.stack([flow_z[:, 1, :, :, :], flow_z[:, 0, :, :, :], 
                          torch.zeros_like(flow_z[:, 0, :, :, :])], dim=1)  # (B, 3, D-1, H/2, W/2)
    coord_z1 = coord_z[:, :, :, ::interp, ::interp] + flow_z  # (B, 3, D-1, H/2, W/2)
    coord_z1_norm = coord_z1 / torch.tensor([W, H, D-1], dtype=coord_z1.dtype, device=coord_z1.device).view(1, 3, 1, 1, 1) * 2 - 1

    embedding_z1_f = F.grid_sample(embedding_z1, coord_z1_norm.permute(0, 2, 3, 4, 1), 
                                   padding_mode='zeros', align_corners=True)  # (B, C, D-1, H/2, W/2)
    embedding_z_f = F.interpolate(embedding_z, scale_factor=(1, 1/interp, 1/interp),
                                  mode='trilinear', align_corners=True)  # (B, C, D-1, H/2, W/2)
    
    coord_mask = (coord_z1[:, 0] >= 0) & (coord_z1[:, 0] < W) & (coord_z1[:, 1] >= 0) & (coord_z1[:, 1] < H)  # (B, D-1, H/2, W/2)
    coord_mask = coord_mask.unsqueeze(1)  # (B, 1, D-1, H/2, W/2)

    if cosine:
        loss = torch.sum((1 - F.cosine_similarity(embedding_z1_f, embedding_z_f, dim=1)) * coord_mask.float())
    else:
        loss = torch.sum(F.relu(torch.abs(embedding_z1_f - embedding_z_f) * coord_mask.float() - delta) ** 2)

    # print(loss.data.cpu().numpy()/B/C/D/H/W)
    loss /= B
    return loss


if __name__ == '__main__':
    embedding = torch.randn((4, 32, 1, 60, 60)).expand(4, 32, 20, 60, 60).to(torch.float32)
    flow = torch.randint(low=0, high=1, size=(4, 2, 20, 60, 60)).to(torch.float32)
    # flow = torch.randint(low=-2, high=3, size=(4, 2, 20, 60, 60)).to(torch.float32)

    import time
    start = time.time()
    loss_slow = motion_loss_fast(embedding, flow, interp=1)
    time_slow = time.time() - start
    loss = motion_loss_fast_2(embedding, flow)
    time_fast = time.time() - start - time_slow
    print(loss_slow, time_slow)
    print(loss, time_fast)

    # time_slow = 0
    # num = 10
    # for i in range(num):
    #     start = time.time()
    #     embedding = torch.randn((4, 32, 20, 60, 60))
    #     flow = torch.randint(low=-2, high=2, size=(4, 2, 20, 60, 60))
    #     loss = motion_loss(embedding, flow)
    #     time_slow += time.time() - start
    # print(time_slow / num)
    # time_smooth = 0
    # for i in range(num):
    #     start = time.time()
    #     embedding = torch.randn((4, 32, 20, 60, 60))
    #     loss = smooth_loss(embedding)
    #     time_smooth += time.time() - start
    # print(time_smooth / num)