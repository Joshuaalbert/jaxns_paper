import argparse
import shutil
from timeit import default_timer
import jax.scipy.linalg as jax_linalg
from jax.scipy.special import logsumexp as jax_logsumexp
from jax import numpy as jnp, vmap, random
import pylab as plt
import numpy as np
from scipy.special import logsumexp as np_logsumexp, ndtri
import scipy.linalg as scipy_linalg
from jaxns.plotting import plot_cornerplot, plot_diagnostics, add_colorbar_to_axes
from jaxns.likelihood_samplers.ellipsoid_utils import ellipsoid_clustering, bounding_ellipsoid, ellipsoid_params
from jax.config import config

config.update("jax_enable_x64", True)


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



def plot_log_likelihood(log_likelihood, plot_ellipsoids = True):

    theta1 = jnp.linspace(-5, 15, 500)
    T1, T2 = jnp.meshgrid(theta1, theta1, indexing='ij')
    theta1 = jnp.stack([T1.flatten(), T2.flatten()], axis=1)
    lik = vmap(log_likelihood)(theta1).reshape((500, 500))
    fig, ax = plt.subplots(1,1,figsize=(5,4))
    ax.imshow(jnp.exp(lik).T, cmap='bone_r', origin='lower', extent=(-5,15,-5,15))
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.grid()
    add_colorbar_to_axes(ax, 'bone_r', vmin=0., vmax=jnp.max(jnp.exp(lik)),
                         label=r"$\mathcal{L}(\theta_1, \theta_2)$")
    plt.tight_layout()
    fig.savefig('gaussian_mixture_log_likelihood.pdf')

    if plot_ellipsoids:
        select = jnp.where(lik.flatten() > jnp.percentile(lik.flatten(), 95))[0]
        keep = random.shuffle(random.PRNGKey(345326),select)[:1000]
        ax.scatter(theta1[keep,0], theta1[keep,1], c='red',marker='.', s=1, label='samples', alpha=0.5)
        ax.legend()

        log_VS = jnp.log(select.size) - jnp.log(lik.size)
        points = theta1[keep, :]
        depth=7
        K = 2**(depth-1)

        cluster_id, ellipsoid_parameters = ellipsoid_clustering(random.PRNGKey(324532), points, depth, log_VS=log_VS)
        mu, C = vmap(lambda k: bounding_ellipsoid(points, cluster_id == k))(jnp.arange(K))
        radii, rotation = vmap(ellipsoid_params)(C)

        theta = jnp.linspace(0., jnp.pi * 2, 100)
        x = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=0)

        for i, (mu, radii, rotation) in enumerate(zip(mu, radii, rotation)):
            y = mu[:, None] + rotation @ jnp.diag(radii) @ x
            ax.plot(y[0, :], y[1, :], c=plt.cm.jet(i / K), lw=0.5)
            mask = cluster_id == i
            # plt.scatter(points[mask, 0], points[mask, 1], c=jnp.atleast_2d(plt.cm.jet(i / K)))

    plt.tight_layout()
    fig.savefig('gaussian_mixture_log_likelihood_with_ellipses.pdf')
    plt.show()


def main(ndims, num_live_points, K, do_dynesty, do_polychord, do_multinest, do_jaxns):
    prior_mu = np.zeros(ndims)
    prior_cov = K**2 * np.diag(np.ones(ndims)) ** 2

    data_mu = [np.zeros(ndims) + k for k in range(K)]
    jax_data_mu = jnp.stack(data_mu, axis=0)
    data_cov1 = np.diag(np.where(np.arange(ndims)==0, 0.5+0.49, 0.5-0.49))

    data_cov2 = np.diag(np.where(np.arange(ndims)==0, 0.5-0.49, 0.5+0.49))

    true_logZ = -np.inf
    for muk in data_mu:
        true_logZ = np.logaddexp(true_logZ,
                                 np.logaddexp(
                                     np_log_normal(muk, 0., data_cov1+prior_cov),
                                     np_log_normal(muk, 0., data_cov2 + prior_cov)))
    true_logZ = true_logZ - np.log(2*K)
    print("True logZ={}".format(true_logZ))

    def np_log_likelihood(theta, **kwargs):
        log_prob = -np.inf
        # log(a (e^x + e^y)) = log(a) + logaddexp(x, y)
        for muk in data_mu:
            log_prob = np.logaddexp(log_prob,
                                    np.logaddexp(
                                        np_log_normal(theta, muk, data_cov1),
                                        np_log_normal(theta, muk, data_cov2)))
        return log_prob - np.log(2*K)

    def jax_log_likelihood(theta, **kwargs):
        log_prob1 = vmap(lambda muk: jax_log_normal(theta, muk, data_cov1))(jax_data_mu)
        log_prob2 = vmap(lambda muk: jax_log_normal(theta, muk, data_cov2))(jax_data_mu)
        log_prob = jnp.logaddexp(log_prob1, log_prob2)
        return jax_logsumexp(log_prob, axis=0) - jnp.log(2.*K)

    if ndims == 2:
        print("Plotting the 2D likelihood of the second model.")
        plot_log_likelihood(jax_log_likelihood)

    def run_dynest():
        try:
            import dynesty
        except:
            raise ImportError("Dynesty not installed. Run `pip install dynesty`.")

        # prior transform (iid standard normal prior)
        def prior_transform(u):
            """Transforms our unit cube samples `u` to a standard normal prior."""
            return ndtri(u) * np.sqrt(np.diag(prior_cov)) + prior_mu

        sampler = dynesty.NestedSampler(np_log_likelihood,
                                        prior_transform,
                                        ndims,
                                        nlive=num_live_points,
                                        bound='multi',
                                        sample='slice',
                                        slices=5)
        t0 = default_timer()
        sampler.run_nested(dlogz=0.01)
        res = sampler.results
        run_time = default_timer() - t0
        print("Dynesty result keys: {}".format(list(res.keys())))
        logZ = res['logz'][-1]
        logZerr = res['logzerr'][-1]
        ess = np_logsumexp(res['logwt']) * 2 - np_logsumexp(res['logwt'] * 2)
        num_likelihood_evaluations = np.sum(res['ncall'])
        score = num_likelihood_evaluations / ess
        print(f"Dynesty run time: {run_time}")
        print(f"Dynesty log(Z): {logZ} +- {logZerr}")
        print(f"Dynesty num_likelihood/ESS: {score}")
        return run_time, logZ, logZerr

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
        output = pypolychord.run_polychord(log_likelihood, ndims, 1, settings, prior, dumper)
        run_time = default_timer() - t0
        logZ, logZerr = output.logZ, output.logZerr
        score = 0.

        print("PolyChord result keys: {}".format(output))

        print(f"PolyChord run time: {run_time}")
        print(f"PolyChord log(Z): {logZ} +- {logZerr}")
        print(f"PolyChord num_likelihood/ESS: {score}")
        return run_time, logZ, logZerr

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

        # run MultiNest
        t0 = default_timer()
        result = solve(LogLikelihood=np_log_likelihood,
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
        return run_time, logZ, logZerr

    def run_jaxns():
        try:
            from jaxns.nested_sampling import NestedSampler
            from jaxns.prior_transforms import PriorChain, UniformPrior, NormalPrior
        except:
            raise ImportError("Install JaxNS!")
        from timeit import default_timer
        from jax import random, jit
        import jax.numpy as jnp

        if ndims < 5:
            depth = 7
        elif ndims == 5:
            depth = 8
        elif ndims == 6:
            depth = 9
        else:
            depth = 9


        prior_transform = PriorChain().push(NormalPrior('theta', prior_mu, jnp.sqrt(jnp.diag(prior_cov))))
        ns = NestedSampler(jax_log_likelihood, prior_transform, sampler_name='slice')

        def run_with_n(n):
            @jit
            def run(key):
                return ns(key=key,
                          num_live_points=n,
                          max_samples=1e8,
                          collect_samples=False,
                          termination_frac=0.001,
                          sampler_kwargs=dict(depth=depth, num_slices=5))

            results = run(random.PRNGKey(0))
            results.logZ.block_until_ready()
            t0 = default_timer()
            results = run(random.PRNGKey(132624))
            results.logZ.block_until_ready()
            run_time = (default_timer() - t0)
            logZ, logZerr = results.logZ, results.logZerr
            score = results.num_likelihood_evaluations/results.ESS

            print('Number of samples taken: {}'.format(results.num_samples))
            print(f"JAXNS run time: {run_time}")
            print(f"JAXNS log(Z): {logZ} +- {logZerr}")
            print(f"JAXNS num_likelihood/ESS: {score}")

            return run_time, logZ, logZerr

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
        run_data.append(run_dynest())
    else:
        try:
            run_data.append((np.load(file_name)['run_data'][0, 0],
                             np.load(file_name)['run_data'][1, 0],
                             np.load(file_name)['run_data'][2, 0]))
        except:
            run_data.append((np.nan, np.nan, np.nan))
    names.append("PolyChord")
    if do_polychord:
        run_data.append(run_polychord())
    else:
        try:
            run_data.append((np.load(file_name)['run_data'][0, 1],
                             np.load(file_name)['run_data'][1, 1],
                             np.load(file_name)['run_data'][2, 1]))
        except:
            run_data.append((np.nan, np.nan, np.nan))
    names.append("MultiNest")
    if do_multinest:
        run_data.append(run_multinest())
    else:
        try:
            run_data.append((np.load(file_name)['run_data'][0, 2],
                             np.load(file_name)['run_data'][1, 2],
                             np.load(file_name)['run_data'][2, 2]))
        except:
            run_data.append((np.nan, np.nan, np.nan))
    names.append('JaxNS')
    if do_jaxns:
        run_data.append(run_jaxns())
    else:
        try:
            run_data.append((np.load(file_name)['run_data'][0, 3],
                             np.load(file_name)['run_data'][1, 3],
                             np.load(file_name)['run_data'][2, 3]))
        except:
            run_data.append((np.nan, np.nan, np.nan))
    run_data = np.array(run_data)
    run_time, logZ, logZerr = run_data.T

    np.savez(file_name, run_data=run_data.T, true_logZ=true_logZ)
    plt.bar(names, run_time, fc="none", ec='black', lw=3.)
    plt.xlabel("Nested sampling package")
    plt.ylabel("Execution time (s)")
    plt.yscale('log')
    plt.savefig(f"{ndims}D_n{num_live_points}_speed_test.png")
    plt.savefig(f"{ndims}D_n{num_live_points}_speed_test.pdf")

def add_args(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument('--ndims', help='number of problem dimensions.',
                        default=2, type=int, required=False)
    parser.add_argument('--num_live_points', help='Number of live points.',
                        default=1000, type=int, required=False)
    parser.add_argument('--K', help='Number of mixture components.',
                        default=10, type=int, required=False)
    parser.add_argument('--do_dynesty', help='Whether to do dynesty run.',
                        default=False, type="bool", required=False)
    parser.add_argument('--do_polychord', help='Whether to do polychord run.',
                        default=False, type="bool", required=False)
    parser.add_argument('--do_multinest', help='Whether to do multinest run.',
                        default=False, type="bool", required=False)
    parser.add_argument('--do_jaxns', help='Whether to do JAXNS run.',
                        default=True, type="bool", required=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Runs a single experiment of the second problem in JAXNS paper.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    print("Running with:")
    for option, value in vars(flags).items():
        print("    {} -> {}".format(option, value))

    main(**vars(flags))