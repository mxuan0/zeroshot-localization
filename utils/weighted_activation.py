import numpy as np

def weighted_activation(activations, weights, base_score, 
                        cap=True, 
                        shift=True, 
                        softmax=True):
  '''
  mode:
    cap: does not include activations that have lower weights than base_score
    shift: subtract base_score from weights
    softmax: perform softmax on scores, should be required if shift and not cap
  '''

  if cap:
    if not (weights > base_score).any():
      print('None of the weights is larger than base')
      return
    
    indices = (weights > base_score).nonzero()
    activations = activations[indices]
    weights = weights[indices]

  if shift:
    weights = weights - base_score

  if softmax:
    weights = np.exp(weights)/np.sum(np.exp(weights))
  
  weighted = activations * np.expand_dims(weights, axis=(1,2))

  return np.maximum(weighted.sum(axis=0), 0)
