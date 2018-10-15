
import causaldag as cd
import save_utils
import os
from multiprocessing import cpu_count
from multiprocessing import Pool
from config import DATA_FOLDER
import numpy as np
from more_itertools import chunked
from collections import defaultdict
from tqdm import tqdm
import argparse
import random

parser = argparse.ArgumentParser('Generate DAGs, CPDAGs, and MECs')

np.random.seed(1729)
random.seed(1729)

parser.add_argument('--densities', type=float, nargs='+', help='Density of graphs (probability of each edge being present)')
parser.add_argument('--nnodes', type=float, nargs='+', help='Sizes of graphs (number of nodes)')
parser.add_argument('--ndags', type=int, help='Number of DAGs per setting')

args = parser.parse_args()

node_sizes = args.nnodes
ndags = args.ndags
densities = args.densities


def get_dags_folder(p, s):
    return os.path.join(DATA_FOLDER, 's=%s/p=%s' % (s, p))


# === THESE HAVE TO BE DEFINED AS SEPARATE FUNCTIONS TO BE USED BY THE MULTIPROCESSING POOL
def get_cpdag(dag):
    return dag.cpdag()


def get_mec_size(cpdag):
    return len(cpdag.all_dags())


def get_optimal_icpdag(dag, cpdag, num_interventions):
    ivs, icpdags = dag.optimal_intervention_greedy(cpdag=cpdag, num_interventions=num_interventions)
    return icpdags


def get_icpdag(dag, cpdag, ivs):
    return dag.interventional_cpdag(ivs, cpdag=cpdag)


def get_num_ivs_to_orient(dag, cpdag):
    ivs, icpdags = dag.fully_orienting_interventions_greedy(cpdag=cpdag)
    return len(ivs)


def save_batch(batch_num, dags_at_once, p, s, run_folder):
    dags = cd.rand.directed_erdos(p, s, size=dags_at_once)
    for i, d in enumerate(dags):
        dag_num = batch_num * dags_at_once + i
        dag_folder = os.path.join(run_folder, 'dag%d' % dag_num)
        os.makedirs(dag_folder)
        dag_amat, node_list = d.to_amat(mode='numpy')
        np.save(os.path.join(dag_folder, 'dag.npy'), dag_amat)


def save_dags(p, s, ndags):
    """
    Save ndags orderDAGs with p nodes and density s in the folder s=s/p=p
    """
    print('=== s=%s, p=%s ===' % (s, p))
    run_folder = get_dags_folder(p, s)
    if not os.path.exists(run_folder):
        os.makedirs(run_folder)
        settings = {
            'nnodes': p,
            'density': s,
            'ndags': ndags
        }
        save_utils.save_yaml(settings, os.path.join(run_folder, 'settings.yml'))

    dags_at_once = 1000

    nbatches = int(np.ceil(ndags/dags_at_once))
    with Pool(cpu_count() - 1) as pool:
        pool.starmap(save_batch, zip(range(nbatches), [dags_at_once]*nbatches, [p]*nbatches, [s]*nbatches, [run_folder]*nbatches))


def save_mec_info(p, s, ndags, num_interventions_list):
    print('=== s=%s, p=%s ===' % (s, p))
    # === CREATE FOLDER AND SAVE SETTINGS
    run_folder = get_dags_folder(p, s)

    # === GENERATE DAGS, CPDAGS, AND ICPDAGS
    ndags_per_batch = cpu_count() - 1
    dag_nums_per_batch = list(chunked(range(ndags), ndags_per_batch))
    mec_sizes_dict = defaultdict(list)
    num_unoriented_dict = defaultdict(list)
    num_ivs_to_orient = []
    
    for dag_nums in tqdm(dag_nums_per_batch):
        dag_amats = [np.load(os.path.join(run_folder, 'dag%d' % dag_num, 'dag.npy')) for dag_num in dag_nums]
        dags = [cd.DAG.from_amat(dag_amat) for dag_amat in dag_amats]
        with Pool(ndags_per_batch) as pool:
            cpdags = pool.map(get_cpdag, dags)
            
            mec_sizes = pool.map(get_mec_size, cpdags)
            mec_sizes_dict[0].extend(mec_sizes)
            num_unoriented_dict[0].extend([len(c.edges) for c in cpdags])

            optimal_icpdags_per_dag = pool.starmap(get_optimal_icpdag, zip(dags, cpdags, [max(num_interventions_list)]*len(dags)))
            for k in num_interventions_list:
                icpdags_k = [optimal_icpdags[k-1] for optimal_icpdags in optimal_icpdags_per_dag]
                mec_sizes_dict[k].extend(pool.map(get_mec_size, icpdags_k))
                num_unoriented_dict[k].extend([len(c.edges) for c in icpdags_k])

            num_ivs_to_orient.extend(pool.starmap(get_num_ivs_to_orient, zip(dags, cpdags)))

        # === SAVE DAGS, CPDAGS, AND ICPDAGS
        for dag_num, d, c in zip(dag_nums, dags, cpdags):
            dag_amat, node_list = d.to_amat(mode='numpy')
            dag_folder = os.path.join(run_folder, 'dag%d' % dag_num)
            cpdag_amat, _ = c.to_amat(node_list, mode='numpy')
            np.save(os.path.join(dag_folder, 'cpdag.npy'), cpdag_amat)

    # === SAVE NUMBER OF UNORIENTED EDGES IN EACH CPDAG AND ICPDAG    
    for k, num_unoriented in num_unoriented_dict.items():
        save_utils.save_list(num_unoriented, os.path.join(run_folder, 'k=%s_num_unoriented.txt' % k))

    # === SAVE SIZE OF MEC FOR EACH CPDAG AND ICPDAG
    for k, mec_sizes in mec_sizes_dict.items():
        save_utils.save_list(mec_sizes, os.path.join(run_folder, 'k=%s_mec_sizes.txt' % k))

    # === SAVE NUMBER INTERVENTIONS NEEDED TO FULLY ORIENT
    save_utils.save_list(num_ivs_to_orient, os.path.join(run_folder, 'num_ivs_to_orient.txt'))
    

for density in densities:
    for p in node_sizes:
        save_dags(p, density, ndags)


for density in densities:
    for p in node_sizes:
        save_mec_info(p, density, ndags, [1, 2])



