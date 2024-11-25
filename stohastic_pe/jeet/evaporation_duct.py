from experiments.stohastic_pe.jeet.utils import get_modified_evaporation_duct
from rwp.sspade import *
from rwp.vis import *
from scipy.special import roots_hermite

environment = Troposphere()
environment.terrain = Terrain(ground_material=SaltWater())

#logging.basicConfig(level=logging.DEBUG)


def get_log_field(freq_hz, antenna_height_m, duct_height_m, max_range_m, max_height_m):
    #print('get_log_field')
    antenna = GaussAntenna(freq_hz=freq_hz, height=antenna_height_m, beam_width=2, elevation_angle=0, polarz='H')

    params = RWPSSpadeComputationalParams(
        max_range_m=max_range_m,
        max_height_m=max_height_m,
        dx_m=100,  # output grid steps affects only on the resulting field, NOT the computational grid
        dz_m=1,
        #dx_computational_grid_wl=2000.0,
        #dz_computational_grid_wl=1.8,
        storage=PickleStorage()
    )

    if freq_hz == 10E9:
        params.dx_computational_grid_wl = 2000.0
        params.dz_computational_grid_wl = 1.8

    if freq_hz == 300E6:
        params.dx_computational_grid_wl = 150.0
        params.dz_computational_grid_wl = 0.3

    if freq_hz == 1E9:
        params.dx_computational_grid_wl = 500.0
        params.dz_computational_grid_wl = 0.33333

    if freq_hz == 3E9:
        params.dx_computational_grid_wl = 1500.0
        params.dz_computational_grid_wl = 1.0

    environment.M_profile = get_modified_evaporation_duct(duct_height_m, 99)
    field = rwp_ss_pade(antenna=antenna, env=environment, params=params)
    field.field = 20*np.log10(np.abs(field.field+1e-16))
    field.field = field.field[:, field.z_grid <= 100]
    field.z_grid = field.z_grid[field.z_grid <= 100]
    field.log10 = True

    return field


def do_model_n(freq_hz, antenna_height_m, duct_expected_height_m, duct_sigma_m, max_range_m, max_height_m, n=10):
    x, w = roots_hermite(n)
    xs = fm.sqrt(2) * duct_sigma_m * x + duct_expected_height_m
    w_p = w[xs > 0]
    xs_p = xs[xs > 0]
    f_x = [get_log_field(freq_hz, antenna_height_m, h, max_range_m, max_height_m) for h in xs_p]
    fields_x = [f.field for f in f_x]
    expected = 1 / fm.sqrt(fm.pi) * sum([a * b for a, b in zip(fields_x, w_p)])
    expected2 = 1 / fm.sqrt(fm.pi) * sum([a**2 * b for a, b in zip(fields_x, w_p)])
    variance = expected2 - expected**2
    sigma_f = np.sqrt(variance)

    mid_field = get_log_field(freq_hz, antenna_height_m, duct_expected_height_m, max_range_m, max_height_m)
    expected_field = deepcopy(mid_field)
    expected_field.field = expected
    error_field = deepcopy(mid_field)
    error_field.field = mid_field.field - expected_field.field
    field_sigma = deepcopy(mid_field)
    field_sigma.field = sigma_f

    return mid_field, expected_field, error_field, field_sigma


@dataclass
class Result:
    mid_field: Field
    expected_field: Field
    error_field: Field
    field_sigma: Field
    title: str


def do_model(freq_hz, antenna_height_m, duct_expected_height_m, duct_sigma_m, max_range_m, max_height_m, title):
    print(title)
    _, _, _, field_sigma_prev = (
        do_model_n(freq_hz, antenna_height_m, duct_expected_height_m, duct_sigma_m, max_range_m, max_height_m, 3))
    for i in range(2, 10):
        n = 2**i + 1
        print(n)
        mid_field, expected_field, error_field, field_sigma = (
            do_model_n(freq_hz, antenna_height_m, duct_expected_height_m, duct_sigma_m, max_range_m, max_height_m, n))
        relerr = np.linalg.norm(field_sigma_prev.field[10:, -1] - field_sigma.field[10:, -1]) / np.linalg.norm(field_sigma_prev.field[10:, -1])
        print(f"relerr = {relerr}")
        if relerr < 0.1:
            return Result(mid_field=mid_field, expected_field=expected_field, error_field=error_field, field_sigma=field_sigma, title=title)
        field_sigma_prev = field_sigma

    return None


def show(model_res: List[Result]):
    fig = plt.figure(figsize=(9, 2.5*len(model_res)), constrained_layout=True)

    subfigs = fig.subfigures(nrows=len(model_res), ncols=1, height_ratios=[1]*(len(model_res)-1) + [1.3])
    for row, subfig in enumerate(subfigs):
        subfig.suptitle(model_res[row].title)
        ax = subfig.subplots(nrows=1, ncols=4)

        norm = Normalize(-80, 0)
        field = model_res[row].mid_field
        extent = [field.x_grid[0]*1E-3, field.x_grid[-1]*1E-3, field.z_grid[0], field.z_grid[-1]]
        im = ax[0].imshow(field.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
        if row == len(model_res)-1:
            subfig.colorbar(im, ax=ax[0], fraction=0.046, location='bottom')
            ax[0].set_xlabel("Range (km)", fontsize=13)
        else:
            ax[0].set_xticklabels([])
        ax[0].set_ylabel("Height (m)", fontsize=13)
        if row == 0:
            ax[0].set_title("L, dB", fontsize=13)
        ax[0].grid(True)

        norm = Normalize(-80, 0)
        field = model_res[row].mid_field
        extent = [field.x_grid[0]*1E-3, field.x_grid[-1]*1E-3, field.z_grid[0], field.z_grid[-1]]
        im = ax[1].imshow(field.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('jet'))
        if row == len(model_res) - 1:
            subfig.colorbar(im, ax=ax[1], fraction=0.046, location='bottom')
            ax[1].set_xlabel("Range (km)", fontsize=13)
        else:
            ax[1].set_xticklabels([])
        if row == 0:
            ax[0].set_title("L, dB", fontsize=13)
        if row == 0:
            ax[1].set_title("E[L], dB", fontsize=13)
        ax[1].set_yticklabels([])
        ax[1].grid(True)

        norm = Normalize(-20, 20)
        field = model_res[row].error_field
        extent = [field.x_grid[0]*1E-3, field.x_grid[-1]*1E-3, field.z_grid[0], field.z_grid[-1]]
        im = ax[2].imshow(field.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('seismic'))
        if row == len(model_res) - 1:
            subfig.colorbar(im, ax=ax[2], fraction=0.046, location='bottom')
            ax[2].set_xlabel("Range (km)", fontsize=13)
        else:
            ax[2].set_xticklabels([])
        ax[2].set_yticklabels([])
        # ax[2].set_ylabel("Height (m)")
        if row == 0:
            ax[2].set_title("L - E[L]")
        ax[2].grid(True)

        norm = Normalize(0, 20)
        field = model_res[row].field_sigma
        extent = [field.x_grid[0]*1E-3, field.x_grid[-1]*1E-3, field.z_grid[0], field.z_grid[-1]]
        im = ax[3].imshow(field.field.T[::-1, :], extent=extent, norm=norm, aspect='auto', cmap=plt.get_cmap('binary'))
        if row == len(model_res) - 1:
            subfig.colorbar(im, ax=ax[3], fraction=0.046, location='bottom')
            ax[3].set_xlabel("Range (km)", fontsize=13)
        else:
            ax[3].set_xticklabels([])
        if row == 0:
            ax[3].set_title("SD")
        ax[3].set_yticklabels([])
        ax[3].grid(True)



#### 1 Варьируем СКО высоты В.И.№№№№№№№№№№№№
# r_list = []
# r = do_model(
#     freq_hz=10E9,
#     antenna_height_m=10,
#     duct_expected_height_m=20,
#     duct_sigma_m=0.5,
#     max_range_m=300E3,
#     max_height_m=100,
#     title="σ = 0.5 m"
# )
# r_list += [r]
#
# r = do_model(
#     freq_hz=10E9,
#     antenna_height_m=10,
#     duct_expected_height_m=20,
#     duct_sigma_m=1,
#     max_range_m=300E3,
#     max_height_m=100,
#     title="σ = 1 m"
# )
# r_list += [r]
#
# r = do_model(
#     freq_hz=10E9,
#     antenna_height_m=10,
#     duct_expected_height_m=20,
#     duct_sigma_m=2,
#     max_range_m=300E3,
#     max_height_m=100,
#     title="σ = 2 m"
# )
# r_list += [r]
#
# r = do_model(
#     freq_hz=10E9,
#     antenna_height_m=10,
#     duct_expected_height_m=20,
#     duct_sigma_m=3,
#     max_range_m=300E3,
#     max_height_m=100,
#     title="σ = 3 m"
# )
# r_list += [r]
#
# r = do_model(
#     freq_hz=10E9,
#     antenna_height_m=10,
#     duct_expected_height_m=20,
#     duct_sigma_m=4,
#     max_range_m=300E3,
#     max_height_m=100,
#     title="σ = 4 m"
# )
# r_list += [r]
#
# show(r_list)
# plt.savefig('1.eps')

#### 2 варьируем частоту ####

r_list = []
r = do_model(
    freq_hz=300E6,
    antenna_height_m=10,
    duct_expected_height_m=20,
    duct_sigma_m=2,
    max_range_m=300E3,
    max_height_m=300,
    title='f = 300 MHz'
)
r_list += [r]

r = do_model(
    freq_hz=1E9,
    antenna_height_m=10,
    duct_expected_height_m=20,
    duct_sigma_m=2,
    max_range_m=300E3,
    max_height_m=100,
    title='f = 1 GHz'
)
r_list += [r]

r = do_model(
    freq_hz=3E9,
    antenna_height_m=10,
    duct_expected_height_m=20,
    duct_sigma_m=2,
    max_range_m=300E3,
    max_height_m=100,
    title='f = 3 GHz'
)
r_list += [r]

r = do_model(
    freq_hz=10E9,
    antenna_height_m=10,
    duct_expected_height_m=20,
    duct_sigma_m=2,
    max_range_m=300E3,
    max_height_m=100,
    title='f = 10 GHz'
)
r_list += [r]

show(r_list)
plt.savefig('2.eps')

#### 3 варьируем высоту В.И. ####
# r_list = []
# r = do_model(
#     freq_hz=10E9,
#     antenna_height_m=10,
#     duct_expected_height_m=10,
#     duct_sigma_m=2,
#     max_range_m=300E3,
#     max_height_m=100,
#     title='Duct h = 10 m'
# )
# r_list += [r]
#
# r = do_model(
#     freq_hz=10E9,
#     antenna_height_m=10,
#     duct_expected_height_m=20,
#     duct_sigma_m=2,
#     max_range_m=300E3,
#     max_height_m=100,
#     title='Duct h = 20 m'
# )
# r_list += [r]
#
# r = do_model(
#     freq_hz=10E9,
#     antenna_height_m=10,
#     duct_expected_height_m=30,
#     duct_sigma_m=2,
#     max_range_m=300E3,
#     max_height_m=100,
#     title='Duct h = 30 m'
# )
# r_list += [r]
#
# show(r_list)
# plt.savefig('3.eps')
#
# #### 4 варьируем высоту антенны ####
# r_list = []
# r = do_model(
#     freq_hz=10E9,
#     antenna_height_m=5,
#     duct_expected_height_m=20,
#     duct_sigma_m=2,
#     max_range_m=300E3,
#     max_height_m=100,
#     title='Ant. h = 5 m'
# )
# r_list += [r]
#
# r = do_model(
#     freq_hz=10E9,
#     antenna_height_m=30,
#     duct_expected_height_m=20,
#     duct_sigma_m=2,
#     max_range_m=300E3,
#     max_height_m=100,
#     title='Ant. h = 30 m'
# )
# r_list += [r]
#
# r = do_model(
#     freq_hz=10E9,
#     antenna_height_m=60,
#     duct_expected_height_m=20,
#     duct_sigma_m=2,
#     max_range_m=300E3,
#     max_height_m=100,
#     title='Ant. h = 60 m'
# )
# r_list += [r]
#
# show(r_list)
# plt.savefig('4.eps')
