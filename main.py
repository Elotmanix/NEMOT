import torch
import os
from neural_estimation.models import MOT_NE_alg
from utils.config import PreprocessMeta
from sinkhorn.sinkhorn_utils import MOT_Sinkhorn
from utils.data_fns import gen_data
import pprint

def main():
    # TD:
    params = PreprocessMeta()
    # if params.alg in ['ne_mot', 'sinkhorn_mot'] or params.data_dist == 'uniform':
    params.dims = [params.dim]*params.k

    os.environ["CUDA_VISIBLE_DEVICES"] = str(params['cuda_visible'])
    device = torch.device("cuda:0" if (torch.cuda.is_available() and params['device'] == 'gpu') else "cpu")
    print(f"Using {device}")

    print("Simultion Parameters:")
    print("=" * 20)
    for key, value in params.items():
        print(f"{key: <15}: {value}")
    print("=" * 20)

    if params.alg == 'sinkhorn_mot' and params.cost_graph == 'full' and params.n >= 750 and params.dataset != 'mnist':
        params.n = 500
        print(f'n is too big for sinkhorn algorithm. Setting n to {params.n}')
    X = gen_data(params)

    if params.alg == 'ne_mot':
        dim = X.shape[1]
        X = X.to(device)
        MOT_agent = MOT_NE_alg(params, device, dim)
        MOT_agent.train_mot(X)
        # MOT_agent.calc_plan(X)
    elif params.alg == 'sinkhorn_mot':
        (X,MU) = X
        MOT_agent = MOT_Sinkhorn(params, X, MU)
        MOT_agent.solve_sinkhorn()
        ## FOR DEBUG:
        # MOT_agent.calc_emot_from_phi()


    MOT_agent.save_results(X)



if __name__ == '__main__':
    main()
