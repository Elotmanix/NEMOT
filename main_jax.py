import torch
import os
from neural_estimation.models import MOT_NE_alg_JAX
from utils.config import PreprocessMeta
from utils.data_fns import gen_data_JAX

def main():
    # TD:
    params = PreprocessMeta()
    # if params.alg in ['ne_mot', 'sinkhorn_mot'] or params.data_dist == 'uniform':
    params.dims = [params.dim]*params.k

    os.environ["CUDA_VISIBLE_DEVICES"] = str(params['cuda_visible'])

    print("Simultion Parameters:")
    print("=" * 20)
    for key, value in params.items():
        print(f"{key: <15}: {value}")
    print("=" * 20)

    X = gen_data_JAX(params)
    MOT_agent = MOT_NE_alg_JAX(params)
    MOT_agent.train_mot(X)
    MOT_agent.save_results(X)



if __name__ == '__main__':
    main()