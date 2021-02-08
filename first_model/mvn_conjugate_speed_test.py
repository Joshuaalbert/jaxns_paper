import argparse
import shutil
from timeit import default_timer
import jax.scipy.linalg as jax_linalg
from jax import numpy as jnp
import pylab as plt
import numpy as np
from scipy.special import logsumexp, ndtri
import scipy.linalg as scipy_linalg
from jax.config import config

config.update("jax_enable_x64", True)

num_calls = 0

def np_log_normal(x, mean, cov):
    L = np.linalg.cholesky(cov)
    dx = x - mean
    dx = scipy_linalg.solve_triangular(L, dx, lower=True)
    return -0.5 * x.size * np.log(2. * np.pi) - np.sum(np.log(np.diag(L))) \
           - 0.5 * dx @ dx


def jax_log_normal(x, mean, cov):
    L = jnp.linalg.cholesky(cov)
    dx = x - mean
    dx = jax_linalg.solve_triangular(L, dx, lower=True)
    return -0.5 * x.size * jnp.log(2. * jnp.pi) - jnp.sum(jnp.log(jnp.diag(L))) \
           - 0.5 * dx @ dx


def main(ndims, num_live_points, do_dynesty, do_polychord, do_multinest, do_jaxns):
    prior_mu = 2 * np.ones(ndims)
    prior_cov = np.diag(np.ones(ndims)) ** 2

    data_mu = np.zeros(ndims)
    data_cov = np.diag(np.ones(ndims)) ** 2
    data_cov = np.where(data_cov == 0., 0.95, data_cov)

    true_logZ = np_log_normal(data_mu, prior_mu, prior_cov + data_cov)
    print("True logZ={}".format(true_logZ))

    np_log_likelihood = lambda theta, **kwargs: np_log_normal(theta, data_mu, data_cov)
    jax_log_likelihood = lambda theta, **kwargs: jax_log_normal(theta, data_mu, data_cov)

    def run_dynesty():
        try:
            import dynesty
        except:
            raise ImportError("Dynesty not installed. Run `pip install dynesty`.")

        # prior transform (iid standard normal prior)
        def prior_transform(u):
            """Transforms our unit cube samples `u` to a standard normal prior."""
            return ndtri(u) * np.sqrt(np.diag(prior_cov)) + prior_mu

        def counting_function(params):
            counting_function.calls += 1
            return np_log_likelihood(params)

        counting_function.calls = 0

        sampler = dynesty.NestedSampler(counting_function,
                                        prior_transform,
                                        ndims,
                                        nlive=num_live_points,
                                        bound='single',
                                        sample='slice',
                                        slices=5)
        t0 = default_timer()
        sampler.run_nested(dlogz=0.01)
        res = sampler.results
        run_time = default_timer() - t0
        print("Dynesty result keys: {}".format(list(res.keys())))
        logZ = res['logz'][-1]
        logZerr = res['logzerr'][-1]
        ess = logsumexp(res['logwt']) * 2 - logsumexp(res['logwt'] * 2)
        num_likelihood_evaluations = np.sum(res['ncall'])
        score = num_likelihood_evaluations / ess
        print(f"Dynesty run time: {run_time}")
        print(f"Dynesty log(Z): {logZ} +- {logZerr}")
        print(f"Dynesty num_likelihood/ESS: {score}")
        return run_time, logZ, logZerr, counting_function.calls

    def run_polychord():
        try:
            import pypolychord
            from pypolychord.settings import PolyChordSettings
            from pypolychord.priors import UniformPrior, GaussianPrior
        except:
            raise ImportError("Polychord not installed.\n"
                              "Run `git clone https://github.com/PolyChord/PolyChordLite.git\n"
                              "cd PolyChordLite\npython setup.py install`.")

        def log_likelihood(theta):
            """ Simple Gaussian Likelihood"""
            logL = np_log_likelihood(theta)
            return logL, [np.sum(theta)]

        def counting_function(params):
            counting_function.calls += 1
            return log_likelihood(params)

        counting_function.calls = 0


        def prior(hypercube):
            """ Uniform prior from [-1,1]^D. """
            return ndtri(hypercube) * np.sqrt(np.diag(prior_cov)) + prior_mu

        def dumper(live, dead, logweights, logZ, logZerr):
            return

        settings = PolyChordSettings(ndims, 1)
        settings.file_root = 'polychord'
        settings.nlive = num_live_points
        settings.do_clustering = True
        settings.read_resume = False
        settings.num_repeats = 5*ndims

        t0 = default_timer()
        output = pypolychord.run_polychord(counting_function, ndims, 1, settings, prior, dumper)
        run_time = default_timer() - t0
        logZ, logZerr = output.logZ, output.logZerr
        score = 0.

        num_likelihood_evaluations = output.nlike

        print("PolyChord result keys: {}".format(output))

        print(f"PolyChord run time: {run_time}")
        print(f"PolyChord log(Z): {logZ} +- {logZerr}")
        print(f"PolyChord num_likelihood/ESS: {score}")
        return run_time, logZ, logZerr, counting_function.calls

    def run_multinest():
        ### multinest
        try:
            from pymultinest.solve import solve
            from pymultinest.analyse import Analyzer
        except:
            raise ImportError(
                "Multinest is not installed.\nFollow directions on http://johannesbuchner.github.io/PyMultiNest/install.html.")
        import os
        os.makedirs('chains', exist_ok=True)
        prefix = "chains/multinest-"

        # prior transform (iid standard normal prior)
        def prior_transform(u):
            """Transforms our unit cube samples `u` to a standard normal prior."""
            return ndtri(u) * np.sqrt(np.diag(prior_cov)) + prior_mu

        def counting_function(params):
            counting_function.calls += 1
            return np_log_likelihood(params)

        counting_function.calls = 0

        # run MultiNest
        t0 = default_timer()
        result = solve(LogLikelihood=counting_function,
                       Prior=prior_transform,
                       n_dims=ndims,
                       outputfiles_basename=prefix,
                       verbose=False,
                       n_live_points=num_live_points,
                       max_modes=100,
                       evidence_tolerance=0.5,
                       sampling_efficiency=0.3)
        run_time = default_timer() - t0

        # analyser = Analyzer(ndims, outputfiles_basename = prefix)
        # stats = analyser.get_stats()
        logZ, logZerr = result['logZ'], result['logZerr']
        score = 0.
        print("Multinest results:", result)

        print(f"MultiNEST run time: {run_time}")
        print(f"MultiNEST log(Z): {logZ} +- {logZerr}")
        print(f"MultiNEST num_likelihood/ESS: {score}")
        return run_time, logZ, logZerr, counting_function.calls

    def run_jaxns():
        try:
            from jaxns.nested_sampling import NestedSampler
            from jaxns.prior_transforms import PriorChain, UniformPrior, NormalPrior
        except:
            raise ImportError("Install JaxNS!")
        from timeit import default_timer
        from jax import random, jit
        import jax.numpy as jnp


        prior_transform = PriorChain().push(NormalPrior('theta', prior_mu, jnp.sqrt(jnp.diag(prior_cov))))
        ns = NestedSampler(jax_log_likelihood, prior_transform, sampler_name='slice')

        def run_with_n(n):
            @jit
            def run(key):
                return ns(key=key,
                          num_live_points=n,
                          max_samples=1e6,
                          collect_samples=False,
                          termination_frac=0.001,
                          sampler_kwargs=dict(depth=3, num_slices=5))

            results = run(random.PRNGKey(0))
            results.logZ.block_until_ready()
            t0 = default_timer()
            results = run(random.PRNGKey(1))
            results.logZ.block_until_ready()
            run_time = (default_timer() - t0)
            logZ, logZerr = results.logZ, results.logZerr
            score = results.num_likelihood_evaluations/results.ESS

            print(f"JAXNS run time: {run_time}")
            print(f"JAXNS log(Z): {logZ} +- {logZerr}")
            print(f"JAXNS num_likelihood/ESS: {score}")

            return run_time, logZ, logZerr, results.num_samples

        return run_with_n(num_live_points)

    try:
        shutil.rmtree('chains')
    except FileNotFoundError:
        pass


    file_name = f"{ndims}D_n{num_live_points}.npz"
    names = []
    run_data = []
    names.append("Dynesty")
    if do_dynesty:
        run_data.append(run_dynesty())
    else:
        try:
            run_data.append((np.load(file_name)['run_data'][0,0],
                             np.load(file_name)['run_data'][1,0],
                             np.load(file_name)['run_data'][2,0],
                             np.load(file_name)['run_data'][3, 0]
                            ))
        except:
            run_data.append((np.nan, np.nan, np.nan, np.nan))
    names.append("PolyChord")
    if do_polychord:
        run_data.append(run_polychord())
    else:
        try:
            run_data.append((np.load(file_name)['run_data'][0,1],
                             np.load(file_name)['run_data'][1,1],
                             np.load(file_name)['run_data'][2,1],
                             np.load(file_name)['run_data'][3,1])  )
        except:
            run_data.append((np.nan, np.nan, np.nan, np.nan))
    names.append("MultiNest")
    if do_multinest:
        run_data.append(run_multinest())
    else:
        try:
            run_data.append((np.load(file_name)['run_data'][0,2],
                             np.load(file_name)['run_data'][1,2],
                             np.load(file_name)['run_data'][2,2],
                             np.load(file_name)['run_data'][3,2],
                             ))
        except:
            run_data.append((np.nan, np.nan, np.nan, np.nan))
    names.append('JaxNS')
    if do_jaxns:
        run_data.append(run_jaxns())
    else:
        try:
            run_data.append((np.load(file_name)['run_data'][0,3],
                             np.load(file_name)['run_data'][1,3],
                             np.load(file_name)['run_data'][2,3],
                             np.load(file_name)['run_data'][3,3],
                             ))
        except:
            run_data.append((np.nan, np.nan, np.nan, np.nan))
    run_data = np.array(run_data)
    run_time, logZ, logZerr, nlik_calls = run_data.T

    np.savez(file_name, run_data=run_data.T, true_logZ = true_logZ)

    plt.bar(names, run_time, fc="none", ec='black', lw=3.)
    plt.xlabel("Nested sampling package")
    plt.ylabel("Execution time (s)")
    plt.yscale('log')
    plt.savefig(f"{ndims}D_n{num_live_points}_speed_test.png")
    plt.savefig(f"{ndims}D_n{num_live_points}_speed_test.pdf")

    plt.bar(names, run_time, fc="none", ec='black', lw=3.)
    plt.xlabel("Nested sampling package")
    plt.ylabel("Number of likelihood evaluations [1]")
    plt.yscale('log')
    plt.savefig(f"{ndims}D_n{num_live_points}_efficiency_test.png")
    plt.savefig(f"{ndims}D_n{num_live_points}_efficiency_test.pdf")
    # plt.show()

def add_args(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument('--ndims', help='number of problem dimensions.',
                        default=3, type=int, required=False)
    parser.add_argument('--num_live_points', help='Number of live points.',
                        default=1000, type=int, required=False)
    parser.add_argument('--do_dynesty', help='Whether to do dynesty run.',
                        default=False, type="bool", required=False)
    parser.add_argument('--do_polychord', help='Whether to do polychord run.',
                        default=False, type="bool", required=False)
    parser.add_argument('--do_multinest', help='Whether to do multinest run.',
                        default=True, type="bool", required=False)
    parser.add_argument('--do_jaxns', help='Whether to do jaxns run.',
                        default=False, type="bool", required=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Runs a single experiment of first problem for JAXNS paper.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    print("Running with:")
    for option, value in vars(flags).items():
        print("    {} -> {}".format(option, value))

    main(**vars(flags))