import xarray as xr
import os
import pandas as pd
import pytz
from datetime import datetime, timedelta
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from grib_puller import GRIB_DL
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from pandas.plotting import register_matplotlib_converters
import numpy as np
import platform


def main():
    register_matplotlib_converters()
    global imgdir
    imgdir = os.path.join(os.path.sep, 'home', 'smotley', 'images', 'weather_email')
    if platform.system() == 'Windows':
        imgdir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Images'))
    #models = ['gfs', 'nam']
    models = ['gfs']
    today = datetime.today().strftime('%Y%m%d')
    lat_mf, lon_mf = 39.10, -120.388  # lat, lon of Middle Fork area between French Meadows and Hell Hole
    for model in models:
        if model == 'gfs':
            lon_mf = (360 + lon_mf)
        df = model_fz_level(model, lat_mf, lon_mf, today)
        create_plot(df, model)
    return

def model_fz_level(model, lat_mf, lon_mf, date):

    model_res = 'conusnest'
    if model == 'gfs': model_res = '0p25'
    ds = GRIB_DL(model=model, model_run='06z', model_resolution=model_res, date=date)
    #gfs_ds = GRIB_DL(model='gfs', model_resolution='0p25', date='20191230')
    #nn_ds = GRIB_DL(model='nam', model_run='12z', model_resolution='conusnest', date='20191230')

    # Convert interval liquid precip from mm to inches.
    qpf_name = f'qpf_{model}'
    qpf = ds.pull_liquid_precip_intervals(lat=lat_mf, lon=lon_mf, variable='apcpsfc') * 0.03937
    qpf_end = pd.DatetimeIndex(pd.to_datetime(qpf['time'].values, utc=True)).tz_convert('US/Pacific')
    if 'interval_start' in qpf.coords:
        qpf_start = pd.DatetimeIndex(pd.to_datetime(qpf['interval_start'].values, utc=True)).tz_convert('US/Pacific')
    else:
        start_series = pd.Series(qpf_end).shift(1)
        if len(qpf_end) > 1:
            start_series.iloc[0] = qpf_end[0] - (qpf_end[1] - qpf_end[0])
        elif len(qpf_end) == 1:
            start_series.iloc[0] = qpf_end[0] - pd.Timedelta(hours=3)
        qpf_start = pd.DatetimeIndex(start_series.values)

    qpf_start_col = f'qpf_start_{model}'
    df_qpf = pd.DataFrame({qpf_name: qpf.values.astype(float)}, index=qpf_end)
    df_qpf[qpf_start_col] = qpf_start

    hgt_1000 = ds.pull_point_data(lat=lat_mf, lon=lon_mf, level=1000.0, variable='hgtprs') / 10
    hgt_500 = ds.pull_point_data(lat=lat_mf, lon=lon_mf, level=500.0, variable='hgtprs') / 10
    tmp_700 = ds.pull_point_data(lat=lat_mf, lon=lon_mf, level=700.0, variable='tmpprs') - 273.15

    # SL = (thickness in dm + 700 mb temp) * 128 - 64015
    snow_level = (((hgt_500 - hgt_1000) + tmp_700) * 128) - 64015
    df_snowlevel = xarr_to_dataframe(snow_level, f'snowLevel_{model}')

    df = pd.concat([df_qpf[[qpf_name]], df_snowlevel], axis=1)
    # pull_liquid_precip_intervals already returns per-interval precip (not cumulative).
    df[f'qpf_hr_{model}'] = df[qpf_name].fillna(0)

    # Classify each interval's precip by snow level before aggregating to daily.
    snow_level_at_qpf = df_snowlevel.reindex(df_qpf.index, method='nearest')
    snow_interval = np.where(
        snow_level_at_qpf[f'snowLevel_{model}'] <= 5000,
        df_qpf[qpf_name],
        0.0,
    )
    df[f'snow_hr_{model}'] = 0.0
    df.loc[df_qpf.index, f'snow_hr_{model}'] = np.where(
        snow_level_at_qpf[f'snowLevel_{model}'] <= 5000,
        df.loc[df_qpf.index, f'qpf_hr_{model}'],
        0,
    )

    # Split intervals at midnight (local time) so precip spanning days is allocated proportionally
    # to each calendar day instead of being lumped into interval end-time day.
    daily_qpf = split_intervals_to_daily_totals(
        starts=df_qpf[qpf_start_col],
        ends=df_qpf.index.to_series(index=df_qpf.index),
        amounts=df_qpf[qpf_name],
    )
    daily_snow = split_intervals_to_daily_totals(
        starts=df_qpf[qpf_start_col],
        ends=df_qpf.index.to_series(index=df_qpf.index),
        amounts=pd.Series(snow_interval, index=df_qpf.index),
    )
    daily_df = pd.concat(
        [daily_qpf.rename(f'qpf_hr_{model}'), daily_snow.rename(f'snow_hr_{model}')],
        axis=1,
    ).fillna(0.0)
    if len(df_qpf.index) > 0:
        day_start = df_qpf[qpf_start_col].min().normalize()
        day_end = df_qpf.index.max().normalize()
        full_days = pd.date_range(start=day_start, end=day_end, freq='D', tz=df_qpf.index.tz)
        daily_df = daily_df.reindex(full_days, fill_value=0.0)
    df.attrs['daily_df'] = daily_df
    #df = df.add_suffix('_snowLevel')
    #df['Date'] = df.index
    return df

def split_intervals_to_daily_totals(starts, ends, amounts):
    totals = {}
    for start, end, amount in zip(starts, ends, amounts):
        if pd.isna(start) or pd.isna(end) or pd.isna(amount):
            continue
        amount = float(amount)
        if amount <= 0:
            continue
        if end <= start:
            continue

        duration_seconds = (end - start).total_seconds()
        cursor = start
        while cursor < end:
            next_midnight = cursor.normalize() + pd.Timedelta(days=1)
            segment_end = min(end, next_midnight)
            frac = (segment_end - cursor).total_seconds() / duration_seconds
            day_key = cursor.normalize()
            totals[day_key] = totals.get(day_key, 0.0) + (amount * frac)
            cursor = segment_end

    if not totals:
        return pd.Series(dtype=float)
    out = pd.Series(totals).sort_index()
    out.index = pd.DatetimeIndex(out.index)
    return out

def xarr_to_dataframe(ds, name):
    ds.name = name
    df = ds.to_dataframe()
    if name in df.columns:
        df = df[[name]]
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(tz=pytz.utc)
    df.index = df.index.tz_convert('US/Pacific')
    return df

def create_plot(df, model):
    fig, ax1 = plt.subplots(figsize=(14, 8), dpi=180, facecolor='#0b1220')
    ax1.set_facecolor('#111827')
    ax1.patch.set_edgecolor('#334155')
    ax1.patch.set_linewidth(1.2)

    qpf_col = f'qpf_hr_{model}'
    snow_qpf_col = f'snow_hr_{model}'
    snow_level_col = f'snowLevel_{model}'
    daily_df = df.attrs.get('daily_df')
    if daily_df is None:
        # Fallback if daily pre-aggregation was not attached by model_fz_level.
        daily_df = df[[qpf_col, snow_qpf_col]].resample('d').sum()
    else:
        daily_df = daily_df.copy()
    # Shift bar positions to noon so they center on each day.
    daily_df.index = daily_df.index + pd.Timedelta(hours=12)

    xaxis_lowlimit = datetime.now(pytz.timezone('US/Pacific'))
    xaxis_uplimit = datetime.now(pytz.timezone('US/Pacific')) + timedelta(days=9)
    ax1.set_xlim([xaxis_lowlimit, xaxis_uplimit])
    ax1.set_ylim([0.0, 10000])

    snow_line_color = '#7dd3fc'
    snow_fill_color = '#0284c7'
    total_bar_color = '#22c55e'
    cold_bar_color = '#3b82f6'
    text_color = '#f8fafc'
    subtext_color = '#cbd5e1'
    grid_color = '#253247'

    fig.suptitle(
        'Middle Fork Forecast: Snow Level and Liquid Precipitation',
        x=0.06,
        y=0.985,
        ha='left',
        va='top',
        fontsize=24,
        fontweight='bold',
        color=text_color,
    )
    fig.text(
        0.06,
        0.94,
        'Green bars = Total daily precip. Blue bars = Precip that fell when snow level was below 5000 ft',
        ha='left',
        va='top',
        fontsize=13,
        color=subtext_color,
    )

    snow_series = df[snow_level_col].dropna()
    if not snow_series.empty:
        ax1.plot(
            snow_series.index,
            snow_series.values,
            color=snow_line_color,
            linewidth=3.4,
            label='Snow Level (ft)',
            zorder=4,
        )
        ax1.fill_between(
            snow_series.index,
            0,
            snow_series.values,
            color=snow_fill_color,
            alpha=0.20,
            zorder=3,
        )
    else:
        ax1.plot([], [], color=snow_line_color, linewidth=3.4, label='Snow Level (ft)')

    ax1.axhline(5000, color='#64748b', linestyle='--', linewidth=1.2, zorder=2)
    ax1.text(
        xaxis_lowlimit + timedelta(hours=8),
        5160,
        '5000 ft threshold',
        color='#cbd5e1',
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.22', facecolor='#0f172a', edgecolor='#475569', alpha=0.96),
    )

    ax1.set_ylabel('Snow Level (ft)', color=snow_line_color, fontsize=14, fontweight='bold')
    ax1.tick_params(axis='y', colors='#bae6fd', labelsize=12)
    ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax1.grid(axis='y', color=grid_color, linewidth=1.0, alpha=1.0)
    ax1.grid(axis='x', color='#1e293b', linewidth=0.9, alpha=0.9)
    ax1.set_axisbelow(True)

    ax2 = ax1.twinx()
    qpf_axis_top = 4.0
    ax2.set_ylim([0.0, qpf_axis_top])
    ax2.set_ylabel('24-hour Liquid Precip (inches)', color='#86efac', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='y', colors='#86efac', labelsize=12)
    ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.2f}"'))

    qpf_bar = ax2.bar(
        daily_df.index,
        daily_df[qpf_col],
        width=0.82,
        color=total_bar_color,
        alpha=0.88,
        edgecolor='#0f172a',
        linewidth=0.9,
        label='Total Liquid Precip',
        zorder=1,
    )
    sn_qpf_bar = ax2.bar(
        daily_df.index,
        daily_df[snow_qpf_col],
        width=0.52,
        color=cold_bar_color,
        alpha=0.98,
        edgecolor='#0f172a',
        linewidth=0.9,
        label='Precip w/ Snow Level <= 5000 ft',
        zorder=2,
    )

    for dt, total in zip(daily_df.index, daily_df[qpf_col]):
        if dt < xaxis_lowlimit or dt > xaxis_uplimit or total < 0.05:
            continue
        ax2.text(
            dt,
            min(total + (qpf_axis_top * 0.022), qpf_axis_top * 0.985),
            f'{total:.2f}"',
            ha='center',
            va='bottom',
            fontsize=14,
            fontweight='bold',
            color='#f8fafc',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#0f172a', edgecolor='#334155', alpha=0.97),
            zorder=5,
        )

    ax1.xaxis.set_major_locator(mdates.HourLocator(byhour=12))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%a %m/%d"))
    ax1.tick_params(axis='x', labelsize=13, colors=text_color, pad=10)
    for label in ax1.get_xticklabels():
        label.set_fontweight('semibold')

    for spine in ['top']:
        ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)
    ax1.spines['left'].set_color('#64748b')
    ax1.spines['bottom'].set_color('#64748b')
    ax2.spines['right'].set_color('#64748b')

    fig.subplots_adjust(left=0.08, right=0.92, top=0.82, bottom=0.2)
    snow_handle = ax1.lines[0]
    legend = fig.legend(
        handles=[qpf_bar, sn_qpf_bar, snow_handle],
        labels=['Total Liquid Precip', 'Precip w/ Snow Level <= 5000 ft', 'Snow Level (ft)'],
        loc='lower center',
        bbox_to_anchor=(0.5, 0.05),
        ncol=3,
        frameon=True,
        fancybox=True,
        framealpha=0.98,
        borderpad=0.6,
        fontsize=12,
        columnspacing=1.8,
    )
    legend.get_frame().set_facecolor('#0f172a')
    legend.get_frame().set_edgecolor('#334155')
    legend.get_frame().set_linewidth(1.0)
    for txt in legend.get_texts():
        txt.set_color('#e2e8f0')

    plt.savefig(os.path.join(imgdir, 'qpf_graph.png'), dpi=230, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return

def gradient_fill(x, y, fill_color=None, ax=None, **kwargs):
    """
    Plot a line with a linear alpha gradient filled beneath it.
    Adapted From:
    https://stackoverflow.com/questions/29321835/is-it-possible-to-get-color-gradients-under-curve-in-matplotlib
    Parameters
    ----------
    x, y : array-like
        The data values of the line.
    fill_color : a matplotlib color specifier (string, tuple) or None
        The color for the fill. If None, the color of the line will be used.
    ax : a matplotlib Axes instance
        The axes to plot on. If None, the current pyplot axes will be used.
    Additional arguments are passed on to matplotlib's ``plot`` function.

    Returns
    -------
    line : a Line2D instance
        The line plotted.
    im : an AxesImage instance
        The transparent gradient clipped to just the area beneath the curve.
    """
    #x = x.astype(np.int64)
    if ax is None:
        ax = plt.gca()

    line, = ax.plot(x, y, **kwargs)
    if fill_color is None:
        fill_color = line.get_color()

    zorder = line.get_zorder()
    alpha = line.get_alpha()
    alpha = 1.0 if alpha is None else alpha

    z = np.empty((100, 1, 4), dtype=float)
    rgb = mcolors.colorConverter.to_rgb(fill_color)
    z[:,:,:3] = rgb
    z[:,:,-1] = np.linspace(0, alpha, 100)[:,None]

    xmin, xmax, ymin, ymax = x.min(), x.max(), 0, y.max()
    im = ax.imshow(z, aspect='auto', extent=[xmin, xmax, ymin, ymax],
                   origin='lower', zorder=zorder)

    xy = np.column_stack([x, y])
    xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
    clip_path = Polygon(xy, facecolor='none', edgecolor='none', closed=True)
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)

    #ax.autoscale(True)
    date_format = mdates.DateFormatter("%a %m/%d")
    ax.xaxis.set_major_formatter(date_format)

    xaxis_lowlimit = datetime.now(pytz.timezone('US/Pacific'))
    xaxis_uplimit = datetime.now(pytz.timezone('US/Pacific')) + timedelta(days=9)
    ax.set_xlim([xaxis_lowlimit, xaxis_uplimit])

    ax.set_ylabel('Snow Level (ft)', color='blue')
    ax.set_xlabel('Date')
    ax.set_ylim([0.0, 10000])

    return line, im

def gradient_bar(df, xaxis_lowlimit, xaxis_uplimit):
    #data = [(0.05, 7000), (0.25, 6000), (0.5, 5000), (1, 4000), (1.5, 5000), (2, 6000)]
    #dfT = pd.DataFrame(data, columns=['qpf', 'snowlevel'], index=("2019-12-25 00:00", "2019-12-25 01:00", "2019-12-25 02:00",
    #                                                        "2019-12-25 03:00", "2019-12-25 04:00", "2019-12-25 05:00"))
    source_img = Image.new("RGBA", (100, 100))

    color_bar_w = 576-80                           # Width: Obtained by going into GIMP and getting x val at corners of graph
    color_bar_h = 50                               # Height of bar
    qpf_limits = [0.001, 0.25, 0.5]                # Lower, middle, and upper limits of cmap for hrly qpf
    colors = ["lightgreen", "lime", "darkgreen"]   # See: https://matplotlib.org/3.1.0/gallery/color/named_colors.html

    norm = plt.Normalize(min(qpf_limits), max(qpf_limits))
    tuples = list(zip(map(norm, qpf_limits), colors))
    cmap = mcolors.LinearSegmentedColormap.from_list("", tuples)

    colors_fz = ["lightskyblue", "dodgerblue", "navy"]  # See: https://matplotlib.org/3.1.0/gallery/color/named_colors.html
    tuples_fz = list(zip(map(norm, qpf_limits), colors_fz))
    cmap_fz = mcolors.LinearSegmentedColormap.from_list("", tuples_fz)

    # To test what the color bar will look like
    #x, y, c = zip(*np.random.rand(30, 3) * 4 - 2)
    #plt.scatter(x, y, c=c, cmap=cmap, norm=norm)
    #plt.colorbar()
    #plt.show()

    px_cnt = 0
    for row in df.itertuples():
        if row.Index >= xaxis_lowlimit:
            px_cnt += 1

    one_px = int(color_bar_w/px_cnt)
    color_bar = Image.new('RGBA', (color_bar_w, color_bar_h))
    cnt = 0
    for idx, row in enumerate(df.itertuples()):
        qpf = row.qpf_hr_gfs
        sl = row.snowLevel_gfs
        date = row.Index

        if date >= xaxis_lowlimit:
            rgb_fill = tuple(int(i * 255) for i in cmap(norm(qpf))[:-1])
            if sl < 5500:
                rgb_fill = tuple(int(i * 255) for i in cmap_fz(norm(qpf))[:-1])
            draw = ImageDraw.Draw(color_bar)
            x = one_px*cnt
            cnt += 1
            if qpf >= qpf_limits[0]:
                draw.rectangle(((0+x, 0), (one_px+x, color_bar_h)), fill=rgb_fill)
            #draw.rectangle(((0+x, px_size), (px_size+x, px_size*2)), fill=rgb_fill)

            #draw.text((20+x, 0), str(qpf), font=ImageFont.truetype("arial.ttf"))
    color_bar.save(os.path.join(imgdir, 'colorbar.png'), "png")
    bar_graph = Image.open(os.path.join(imgdir, 'qpf_graph.png'))
    bar_graph.paste(color_bar,(80,58))
    bar_graph.save(os.path.join(imgdir, 'new_graph.png'), 'png')

    return

if __name__=="__main__":
    main()
