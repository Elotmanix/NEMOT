import torch
import os
from neural_estimation.models import MOT_NE_alg
from utils.config import PreprocessMeta
from sinkhorn.sinkhorn_utils import MOT_Sinkhorn
from utils.data_fns import gen_data

def main():
    # TD:
    params = PreprocessMeta()

    # params = gen_params()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(params['cuda_visible'])
    device = torch.device("cuda:0" if (torch.cuda.is_available() and params['device'] == 'gpu') else "cpu")
    print(f"Using {device}")
    print(params)

    X = gen_data(params)

    if params.alg == 'ne_mot':
        X = X.to(device)
        MOT_agent = MOT_NE_alg(params, device)
        MOT_agent.train_mot(X)
    elif params.alg == 'sinkhorn_mot':
        assert params.n <= 750, "Error Message: n is too big for sinkhorn algorithm."
        (X,MU) = X
        MOT_agent = MOT_Sinkhorn(params, X, MU)
        MOT_agent.solve_sinkhorn()

    MOT_agent.save_results()



if __name__ == '__main__':
    main()