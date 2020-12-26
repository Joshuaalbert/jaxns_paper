import pylab as plt
from matplotlib.transforms import Affine2D
import numpy as np
import glob

def main():
    data_files = sorted(glob.glob("*D_n*.npz"))
    ndims, num_live_points = [],[]
    run_times, logZs, logZerrs, true_logZs = [],[],[],[]
    for file in data_files:
        print("parsing",file)
        ndims.append(int(file.split("D")[0]))
        num_live_points.append(int(file.split("_n")[-1].split('.')[0]))
        run_time, logZ, logZerr = np.load(file)['run_data']
        run_times.append(run_time)
        logZs.append(logZ)
        logZerrs.append(logZerr)
        true_logZs.append(np.load(file)['true_logZ'])
    ndims = np.array(ndims)
    a = np.argsort(ndims)

    run_times = np.array(run_times)
    true_logZs = np.array(true_logZs)
    logZs = np.array(logZs)
    logZerrs = np.array(logZerrs)

    names = ['Dynesty', 'PolyChord', 'MultiNEST', 'JAXNS']
    colors = ['blue','green','red','black']
    markers = ['o','^', 'v', '*']

    fig, axs = plt.subplots(2,1, sharex=True, figsize=(5,5))
    for i, name in enumerate(names):
        axs[0].plot(ndims[a], run_times[a,i], c=colors[i], marker=markers[i], label=name)
    axs[0].set_yscale('log')
    axs[0].set_ylabel('Run time [s]')
    axs[0].grid()
    axs[0].legend()

    # plt.plot(ndims, true_logZs, ls='dotted',c='black')
    axs[1].hlines(0., np.min(ndims), np.max(ndims), linestyles='dotted', colors='grey')
    transforms = [Affine2D().translate(-0.2, 0.0) + axs[1].transData,
                  Affine2D().translate(-0.1, 0.0) + axs[1].transData,
                  Affine2D().translate(0.0, 0.0) + axs[1].transData,
                  Affine2D().translate(0.1, 0.0) + axs[1].transData]
    for i, name in enumerate(names):
        # plt.plot(ndims, logZs[:,i] - true_logZs, c=colors[i], marker=markers[i], label=name)
        axs[1].errorbar(ndims[a], logZs[a,i] - true_logZs[a], logZerrs[a,i], c=colors[i], marker=markers[i],
                     transform=transforms[i], label=name)
    axs[1].set_ylabel(r'$\Delta \log Z$')
    axs[1].set_xlabel('Prior dimension')
    axs[1].legend()
    plt.tight_layout()
    fig.subplots_adjust(hspace=0)
    plt.savefig('performance_first_model.pdf')
    plt.show()


if __name__ == '__main__':
    main()