import torch, pdb
import numpy as np
from tqdm import tqdm
def get_cam_weights(input_tensor_,
                    model,
                    text_embedding,
                    activations,
                    device,
                    BATCH_SIZE=32):
  with torch.no_grad():
    upsample = torch.nn.UpsamplingBilinear2d(
        size=input_tensor_.shape[-2:])
    if isinstance(activations, np.ndarray):
        activation_tensor = torch.from_numpy(activations)
    elif isinstance(activations, torch.Tensor):
        activation_tensor = activations
    else:
        print('Invalid activation datatype')
        exit(1)
    
    if device == 'cuda':
        torch.cuda.empty_cache()

    activation_tensor = activation_tensor.to(device)
    input_tensor = input_tensor_.to(device)
    zeros = torch.zeros_like(input_tensor).to(device)
    zeros_embeddings = model.encode_image(zeros)
    zeros_embeddings /= zeros_embeddings.norm(dim=-1, keepdim=True)
    zeros_embeddings = zeros_embeddings.cpu().numpy()
    
    base_scores = (zeros_embeddings @ text_embedding.T).diagonal()    

    upsampled = upsample(activation_tensor)
    # print(upsampled[0,10,:])
    if torch.isnan(upsampled).any():
      print('nan in upsampled before normal')
      return
    
    maxs = upsampled.view(upsampled.size(0), upsampled.size(1),
                          -1).max(dim=-1)[0]
    mins = upsampled.view(upsampled.size(0), upsampled.size(1),
                          -1).min(dim=-1)[0]

    maxs, mins = maxs[:, :, None, None], mins[:, :, None, None]
    upsampled = (upsampled - mins) / (maxs - mins + 1e-7)
    if torch.isnan(input_tensor).any():
      print('nan in input_tensor')
      return

    if torch.isnan(upsampled).any():
      print('nan in upsampled after normal')
      return
    
    input_tensors = torch.einsum('blhw,bchw->bclhw', input_tensor, upsampled)
    if torch.isnan(input_tensors).any():
      print('nan in input_tensors')
      return 
    scores = []
    for index, tensor in enumerate(input_tensors): 
        score = []
        for i in tqdm(range(0, tensor.size(0), BATCH_SIZE)):
          torch.cuda.empty_cache()
          batch = tensor[i: i + BATCH_SIZE, :]
          if torch.isnan(batch).any():
            print('nan in batch')
            return
          img_embeddings = model.encode_image(batch)
          img_embeddings /= img_embeddings.norm(dim=-1, keepdim=True)
          img_embeddings = img_embeddings.cpu().numpy()
          
          if np.isnan(img_embeddings).any():
            print('nan in embeddings')
            return
          outputs = img_embeddings @ text_embedding[index].T
          # print("outputs", outputs.shape)
          score.append(outputs)
        scores.append(np.concatenate(score[:])) 
    
    scores = np.concatenate(scores, axis=0)
    scores = scores.reshape((activations.shape[0], activations.shape[1]))
    return scores, base_scores
