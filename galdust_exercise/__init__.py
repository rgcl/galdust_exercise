import multiprocessing as mp
from ctypes import Structure, c_double
from io import BytesIO
from os import path
from multiprocessing.sharedctypes import RawArray
import galdust
import numpy as np
from astropy.io import ascii
from astropy.table import Table
import sys
from glob import glob
import matplotlib.pyplot as plt

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

DIR_NAME = path.dirname(__file__)
container = galdust.dl07


class FilterSpecimen(Structure):
    _fields_ = [('wl', c_double), ('flux', c_double)]


def compute_models(also_fits, outdir):

    print('Initializing the compute_models task')
    print(f'The fits option is {also_fits}\n')

    print('Step 1: Determining all possibilities...')
    gammas = np.logspace(-3, 3, 25)
    param_models = []
    for key in container:
        model_data = container[key]
        [
            param_models.append((
                *model_data,
                gamma
            ))
            for gamma in gammas
        ]

    print(f'\tThere are {len(param_models)} models')

    print('Step 2: Computing the models...')

    with mp.Pool(
            initializer=worker_init,
            initargs=(load_filters(), also_fits, outdir)
    ) as pool:
        results = pool.map(compute_models_worker, param_models)
        results = np.array(results)

        print(f'Step 3: Saving results in {outdir}/models.dat')
        ascii.write(results, path.join(outdir, 'models.dat'), names=[
            'umin',
            'umax',
            'q_pah',
            'model',
            'gamma',
            'filter24',
            'filter70',
            'filter100',
            'filter160',
            'filtr250',
            'filter350',
            'filter500'
        ], overwrite=True)


def ploting_models(outdir):
    files = glob(path.join(outdir, 'model*.fits'))
    with mp.Pool(
            initializer=worker_init,
            initargs=(load_filters(), False, outdir)
    ) as pool:
        pool.map(ploting_models_worker, files)


def compute_models_worker(param_model):
    # we references the model in this way and not using the container is because the current
    # implementation of the container reload the files per each instantiation, and we not want to
    # reload the files per each thread
    model = galdust.dl07spec.DustModel(*param_model)

    if gbl_also_fits:
        Table(model.spectrum(), names=('wavelength', 'flux')).write(
            path.join(
                gbl_outdir,
                f'model_{param_model[0]}_{param_model[1]}_{param_model[2]}_{param_model[5]}.fits'
            )
        , format='fits', overwrite=True)

    return [
        param_model[0],  # umin
        param_model[1],  # umax
        param_model[2],  # q_pah
        param_model[3],  # model
        param_model[5]  # gamma
    ] + [
        model.luminosity_per_wavelength(filter_data) for name, filter_data in gbl_filters.items()
    ]


def ploting_models_worker(model_file):
    table = Table.read(model_file)
    plot_file = f'{model_file[0:-5]}.pdf'

    figure = plt.figure()
    plt.loglog(table['wavelength'], table['flux'])
    plt.title('Model Spectrum')
    plt.xlabel('$\lambda[nm]$')
    plt.ylabel('$L_\lambda$ [W/nm/(kg of H)]')
    plt.minorticks_on()
    figure.savefig(plot_file, format='pdf')
    plt.close(figure)


def worker_init(filters, also_fits, outdir):
    filter24, filter70, filter100, filter160, filtr250, filter350, filter500 = filters

    global gbl_filters
    global gbl_also_fits
    global gbl_outdir

    numpylize = lambda filter: np.array([[specimen.wl, specimen.flux] for specimen in filter]).astype(np.float32)
    gbl_filters = {
        '24': numpylize(filter24),
        '70': numpylize(filter70),
        '100': numpylize(filter100),
        '160': numpylize(filter160),
        '250': numpylize(filtr250),
        '350': numpylize(filter350),
        '500': numpylize(filter500)
    }
    gbl_also_fits = also_fits
    gbl_outdir = outdir


def load_filters():
    filters = []
    for filter_name in '24,70,100,160,250,350,500'.split(','):
        filter_dir = path.join(DIR_NAME, 'data', f'{filter_name}.dat')
        with open(filter_dir, 'r') as file:
            file_text = file.read()
            file_data = np.genfromtxt(BytesIO(file_text.encode()))
            filters.append(
                RawArray(
                    FilterSpecimen,
                    [(row[0], row[1]) for row in file_data]
                )
            )
    return tuple(filters)


def main():
    outdir = path.join(DIR_NAME, 'out');
    if 'plots' in sys.argv:
        ploting_models(outdir)
    else:
        compute_models('fits' in sys.argv, outdir)
