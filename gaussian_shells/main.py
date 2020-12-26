from jaxns.nested_sampling import NestedSampler
from jaxns.prior_transforms import PriorChain, UniformPrior, NormalPrior
from jaxns.plotting import plot_cornerplot, plot_diagnostics, add_colorbar_to_axes
from jaxns.likelihood_samplers.ellipsoid_utils import ellipsoid_clustering, bounding_ellipsoid, ellipsoid_params
from jax import random, jit,vmap
from jax import numpy as jnp
import pylab as plt
from timeit import default_timer


def log_likelihood(theta, **kwargs):
    def log_circ(theta, c, r, w):
        return -0.5*(jnp.linalg.norm(theta - c) - r)**2/w**2 - jnp.log(jnp.sqrt(2*jnp.pi*w**2))

    w1 = jnp.array(0.3)
    r1 = jnp.array(4.)
    c1 = jnp.array([0., 0.])
    return log_circ(theta, c1,r1,w1)

def plot_log_likelihood():
    theta1 = jnp.linspace(-6, 6, 500)
    T1, T2 = jnp.meshgrid(theta1, theta1, indexing='ij')
    theta1 = jnp.stack([T1.flatten(), T2.flatten()], axis=1)
    lik = vmap(log_likelihood)(theta1).reshape((500, 500))
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    ax.imshow(jnp.exp(lik).T, cmap='bone_r', origin='lower', extent=(-6, 6, -6, 6))
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.grid()


    select = jnp.where(lik.flatten() > jnp.percentile(lik.flatten(), 85))[0]
    keep = random.shuffle(random.PRNGKey(345326),select)[:1000]
    ax.scatter(theta1[keep,0], theta1[keep,1], c='red',marker='.', s=1, label='samples')
    ax.legend()


    log_VS = jnp.log(select.size) - jnp.log(lik.size)
    points = theta1[keep, :]
    depth=5
    K = 2**(depth-1)

    cluster_id, ellipsoid_parameters = ellipsoid_clustering(random.PRNGKey(324532), points, depth, log_VS=log_VS)
    mu, C = vmap(lambda k: bounding_ellipsoid(points, cluster_id == k))(jnp.arange(K))
    radii, rotation = vmap(ellipsoid_params)(C)

    theta = jnp.linspace(0., jnp.pi * 2, 100)
    x = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=0)

    for i, (mu, radii, rotation) in enumerate(zip(mu, radii, rotation)):
        y = mu[:, None] + rotation @ jnp.diag(radii) @ x
        ax.plot(y[0, :], y[1, :], c=plt.cm.jet(i / K), lw=0.5)

    add_colorbar_to_axes(ax, 'bone_r', vmin=0., vmax=jnp.max(jnp.exp(lik)),
                         label=r"$\mathcal{L}(\theta_1, \theta_2)$")
    fig.savefig('gaussian_shells.pdf')
    plt.show()

def jaxns_nested_sampling():
    prior_chain = PriorChain() \
        .push(UniformPrior('theta', low=-6. * jnp.ones(2), high=6. * jnp.ones(2)))
    ns = NestedSampler(log_likelihood, prior_chain, sampler_name='slice')

    def run_with_n(n):
        @jit
        def run(key):
            return ns(key=key,
                      num_live_points=n,
                      max_samples=1e5,
                      collect_samples=True,
                      termination_frac=0.01,
                      sampler_kwargs=dict(depth=7, num_slices=3))

        t0 = default_timer()
        results = run(random.PRNGKey(0))
        print("Efficiency", results.efficiency)
        print("Time to run (including compile)", default_timer() - t0)
        t0 = default_timer()
        results = run(random.PRNGKey(1))
        print("Efficiency", results.efficiency)
        print("Time to run (no compile)", default_timer() - t0)
        return results

    results = run_with_n(1000)
    print("logZ = {} +- {}".format(results.logZ, results.logZerr))

    plot_diagnostics(results)
    plot_cornerplot(results)



def main():
    plot_log_likelihood()
    jaxns_nested_sampling()




if __name__ == '__main__':
    main()
