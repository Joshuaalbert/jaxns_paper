import argparse
try:
    from jaxns.nested_sampling import NestedSampler
    from jaxns.prior_transforms import PriorChain, UniformPrior, NormalPrior
except:
    raise ImportError("Install JaxNS with GPU support!")
from timeit import default_timer
import numpy as np
import pylab as plt
from functools import partial
from jax import random, jit, numpy as jnp, devices as jax_devices
import jax.scipy.linalg as jax_linalg
from jax.config import config

config.update("jax_enable_x64", True)

TEST_NDIMS = [10,20,30,40,50,60,70,80,90,100]

def jax_log_normal(x, mean, cov):
    L = jnp.linalg.cholesky(cov)
    dx = x - mean
    dx = jax_linalg.solve_triangular(L, dx, lower=True)
    return -0.5 * x.size * jnp.log(2. * jnp.pi) - jnp.sum(jnp.log(jnp.diag(L))) \
           - 0.5 * dx @ dx


def main(use_gpu):
    devices = jax_devices()
    if use_gpu:
        gpu_dev = None
        for dev in devices:
            if dev.platform == 'gpu':
                gpu_dev = dev
                break
        if gpu_dev is None:
            raise ValueError(f"No valid GPU device among {devices}")
    else:
        cpu_dev = None
        for dev in devices:
            if dev.platform == 'cpu':
                cpu_dev = dev
                break
        if cpu_dev is None:
            raise ValueError(f"No valid CPU device among {devices}")
    def run_with_n(ndims, num_live_points):
        prior_mu = 2 * jnp.ones(ndims)
        prior_cov = jnp.diag(jnp.ones(ndims)) ** 2

        data_mu = jnp.zeros(ndims)
        data_cov = jnp.diag(jnp.ones(ndims)) ** 2
        data_cov = jnp.where(data_cov == 0., 0.95, data_cov)

        jax_log_likelihood = lambda theta, **kwargs: jax_log_normal(theta, data_mu, data_cov)

        prior_transform = PriorChain().push(NormalPrior('theta', prior_mu, jnp.sqrt(jnp.diag(prior_cov))))
        ns = NestedSampler(jax_log_likelihood, prior_transform, sampler_name='slice')

        @partial(jit, backend='gpu' if use_gpu else 'cpu')
        def run(key):
            return ns(key=key,
                      num_live_points=num_live_points,
                      max_samples=1e7,
                      collect_samples=False,
                      termination_frac=0.001,
                      sampler_kwargs=dict(depth=3, num_slices=5))

        # first run to make it compile
        results = run(random.PRNGKey(2345256))
        results.logZ.block_until_ready()
        # now measure time
        t0 = default_timer()
        results = run(random.PRNGKey(3498576))
        results.logZ.block_until_ready()
        run_time = (default_timer() - t0)

        print(f"JAXNS num_live_points={num_live_points} ndims={ndims} run time: {run_time}")
        if results.num_samples >= 1e7:
            raise ValueError("Reached maximum number of samples {}.".format(results.num_samples))
        return run_time

    save_file = 'speed_test_results_{}.npz'.format('GPU' if use_gpu else 'CPU')
    run_times = [run_with_n(ndims, ndims*50) for ndims in TEST_NDIMS]
    np.savez(save_file,
             test_ndims = np.array(TEST_NDIMS),
             run_times=np.array(run_times))

    run_times = np.load(save_file)['run_times']
    test_ndims = np.load(save_file)['test_ndims']

    plt.plot(test_ndims, run_times, label='GPU' if use_gpu else 'CPU')
    plt.legend()
    plt.show()




def add_args(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument('--use_gpu', help='Whether to use GPU or else CPU',
                        default=False, type="bool", required=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Runs a speed test on CPU or GPU.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    print("Running with:")
    for option, value in vars(flags).items():
        print("    {} -> {}".format(option, value))

    main(**vars(flags))