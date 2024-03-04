from matplotlib.colors import Normalize

from propagators.mfp import Measure, SearchArea, bartlett_mfp, mv_mfp
from rwp.kediffraction import *
from rwp.vis import *

def rwp_mfp(measures: List[Measure], fields: List[Field], mfp_func) -> Field:
    np_fields = [f.field for f in fields]
    mfp_field = deepcopy(fields[0])
    mfp_np_field = mfp_func(measures=measures, fields=np_fields)
    mfp_field.field = mfp_np_field
    return mfp_field


def fields_from_measures(measures: List[Measure], env: Troposphere, search_area: SearchArea) -> List[Field]:
    fields = []
    for ind_m, measure in enumerate(measures):
        antenna = PointSource(freq_hz=measure.freq_hz, height_m=measure.height_m)
        shifted_env = deepcopy(env)
        shifted_search_area = deepcopy(search_area)
        for knife_edge in shifted_env.knife_edges:
            knife_edge.range -= measure.x_m
        shifted_search_area.min_x_m -= measure.x_m
        shifted_search_area.max_x_m -= measure.x_m
        kdc = KnifeEdgeDiffractionCalculator(src=antenna, env=shifted_env,
                                             min_range_m=shifted_search_area.min_x_m,
                                             max_range_m=shifted_search_area.max_x_m,
                                             inverse=True
                                             )
        field = kdc.calculate()
        field.x_grid += measure.x_m
        fields.append(field)
        print(f'{ind_m / len(measures) * 100}%')
    return fields


def calc(src: Source, env: Troposphere, range_bounds_m, measures: List[Measure], search_area :SearchArea):
    kdc = KnifeEdgeDiffractionCalculator(src=src, env=env, min_range_m=range_bounds_m[0],
                                         max_range_m=range_bounds_m[1], max_propagation_angle=90)
    fwd_field = kdc.calculate()
    for measure in measures:
        measure.value = fwd_field.value(measure.x_m, measure.height_m)

    fields = fields_from_measures(measures=measures, env=env, search_area=search_area)
    bartlett_mfp_field = rwp_mfp(measures, fields, bartlett_mfp)
    mv_mfp_field = rwp_mfp(measures, fields, mv_mfp)

    show(fwd_field, bartlett_mfp_field, mv_mfp_field, src, measures, env)
    print(f'Bartlett {bartlett_mfp_field.argmax()}')
    print(f'MV {mv_mfp_field.argmax()}')

    return fwd_field, bartlett_mfp_field, mv_mfp_field


def show(fwd_field, bartlett_mfp_field, mv_mfp_field, src: Source, measures: List[Measure],
         env: Troposphere):
    plt.rcParams['font.size'] = '13'
    f, ax = plt.subplots(1, 3, figsize=(9, 3.5), constrained_layout=True)

    field = 10 * np.log10(np.abs(fwd_field.field.T[::-1, :]))
    f_min, f_max = np.min(field), np.max(field)
    norm_fwd = Normalize(f_max-40, f_max)
    extent_fwd = [fwd_field.x_grid[0], fwd_field.x_grid[-1], fwd_field.z_grid[0], fwd_field.z_grid[-1]]
    im = ax[0].imshow(field, extent=extent_fwd, norm=norm_fwd, aspect='auto', cmap=plt.get_cmap('jet'))
    f.colorbar(im, ax=ax[0], fraction=0.046, location='bottom')
    ax[0].set_xlabel("Range (m)", fontsize=13)
    ax[0].set_ylabel("Height (m)", fontsize=13)
    ax[0].set_title("10log(|u|), dB", fontsize=13)
    ax[0].grid(True)
    ax[0].plot(0, src.height_m, '*', color='white')
    for m in measures:
        ax[0].plot(m.x_m, m.height_m, '*', color='black')

    field = 10 * np.log10(np.abs(bartlett_mfp_field.field.T[::-1, :]))
    f_min, f_max = np.min(field), np.max(field)
    norm_bartlett = Normalize(f_max - 5, f_max)
    extent_bartlett = [bartlett_mfp_field.x_grid[0], bartlett_mfp_field.x_grid[-1],
                       bartlett_mfp_field.z_grid[0], bartlett_mfp_field.z_grid[-1]]
    im = ax[1].imshow(field, extent=extent_bartlett, norm=norm_bartlett, aspect='auto', cmap=plt.get_cmap('jet'))
    f.colorbar(im, ax=ax[1], fraction=0.046, location='bottom')
    ax[1].set_xlabel("Range (m)", fontsize=13)
    ax[1].set_yticklabels([])
    ax[1].set_title("Bartlett, dB", fontsize=13)
    ax[1].grid(True)
    x_src, z_src = bartlett_mfp_field.argmax()
    ax[1].plot(x_src, z_src, '*', color='white')

    field = 10 * np.log10(np.abs(mv_mfp_field.field.T[::-1, :]))
    f_min, f_max = np.min(field), np.max(field)
    norm_mv = Normalize(f_max - 5, f_max)
    extent_mv = [mv_mfp_field.x_grid[0], mv_mfp_field.x_grid[-1], mv_mfp_field.z_grid[0],
                 mv_mfp_field.z_grid[-1]]
    im = ax[2].imshow(field, extent=extent_mv, norm=norm_mv, aspect='auto', cmap=plt.get_cmap('jet'))
    f.colorbar(im, ax=ax[2], fraction=0.046, location='bottom')
    ax[2].set_xlabel("Range (m)", fontsize=13)
    ax[2].set_yticklabels([])
    ax[2].set_title("MV, dB", fontsize=13)
    ax[2].grid(True)
    x_src, z_src = bartlett_mfp_field.argmax()
    ax[2].plot(x_src, z_src, '*', color='white')

    for kn in env.knife_edges:
        ax[0].axvline(x=kn.range, ymin=0, ymax=kn.height/fwd_field.z_grid[-1], color='brown')
        if bartlett_mfp_field.x_grid[0] <= kn.range <= bartlett_mfp_field.x_grid[-1]:
            ax[1].axvline(x=kn.range, ymin=0, ymax=kn.height/bartlett_mfp_field.z_grid[-1], color='brown')
        if mv_mfp_field.x_grid[0] <= kn.range <= mv_mfp_field.x_grid[-1]:
            ax[2].axvline(x=kn.range, ymin=0, ymax=kn.height/mv_mfp_field.z_grid[-1], color='brown')

    for a in ax[:]:
        for label in (a.get_xticklabels() + a.get_yticklabels()):
            label.set_fontsize(13)
