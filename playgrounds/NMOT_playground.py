import torch
import os
from utils.config import PreprocessMeta
from utils.data_fns import gen_data
from neural_estimation.models import MOT_NE_alg

"""
First test implementation of NMOT to check no problems with the idea itself
Code flow:
0. define simulation aprameters
1. define data 
2. define neural estimator nets (options - a)k different neural nets or b)weight sharing/one net with learned weigted output)
3. run on example
4. compare with Sinkhorn output (I need a Sinkhorn implementation anyways)
"""
def main():
    # TD:
    params = PreprocessMeta()
    if params.alg in ['ne_mot', 'sinkhorn_mot']:
        params.dims = [params.dim]*params.k
    # params = gen_params()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(params['cuda_visible'])
    device = torch.device("cuda:0" if (torch.cuda.is_available() and params['device'] == 'gpu') else "cpu")
    print(f"Using {device}")
    print(params)

    X = gen_data(params)
    X = X.to(device)

    # d = params['dims'][0]
    # n = params['n']
    #
    # X = torch.from_numpy(
    #     np.random.uniform(-1 / np.sqrt(d), 1 / np.sqrt(d), (n, d))).float().to(device)  # uniform [-1/sqrt(d),1/sqrt(d)]^d
    # Y = torch.from_numpy(np.random.uniform(-1 / np.sqrt(d), 1 / np.sqrt(d), (n, d))).float().to(device)
    # X = torch.stack([X,Y], dim=-1)

    if params['alg'] == 'ne_mot':
        MOT_agent = MOT_NE_alg(params, device)
        MOT_agent.train_mot(X)
        MOT_agent.save_results()





########


















if __name__ == '__main__':
    main()