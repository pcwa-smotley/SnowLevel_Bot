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
    qpf = ds.pull_liquid_precip_intervals(lat=lat_mf, lon=lon_mf, variable='apcpsfc') * 0.03937
    df_qpf = xarr_to_dataframe(qpf, f'qpf_{model}')

    hgt_1000 = ds.pull_point_data(lat=lat_mf, lon=lon_mf, level=1000.0, variable='hgtprs') / 10
    hgt_500 = ds.pull_point_data(lat=lat_mf, lon=lon_mf, level=500.0, variable='hgtprs') / 10
    tmp_700 = ds.pull_point_data(lat=lat_mf, lon=lon_mf, level=700.0, variable='tmpprs') - 273.15

    # SL = (thickness in dm + 700 mb temp) * 128 - 64015
    snow_level = (((hgt_500 - hgt_1000) + tmp_700) * 128) - 64015
    df_snowlevel = xarr_to_dataframe(snow_level, f'snowLevel_{model}')

    df = pd.concat([df_qpf, df_snowlevel], axis=1)
    # This is interval liquid precip, so use it directly for bar totals.
    df[f'qpf_hr_{model}'] = df[f'qpf_{model}']
    df[f'qpf_hr_{model}'] = df[f'qpf_hr_{model}'].fillna(0)

    # Align snow-level values to precip timestamps before classifying rain vs snow precip.
    snow_level_at_qpf = df_snowlevel.reindex(df_qpf.index, method='nearest')
    df[f'snow_hr_{model}'] = 0.0
    df.loc[df_qpf.index, f'snow_hr_{model}'] = np.where(
        snow_level_at_qpf[f'snowLevel_{model}'] <= 5000,
        df_qpf[f'qpf_{model}'],
        0,
    )
    #df = df.add_suffix('_snowLevel')
    #df['Date'] = df.index
    return df

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
    fig, ax1 = plt.subplots()
    plt.title('Precipitation and Snow Level Forecast: Middle Fork')

    color = 'tab:blue'
    daily_df = df.resample('d').sum(numeric_only=True)
    date_format = mdates.DateFormatter("%a %m/%d")
    ax1.xaxis.set_major_formatter(date_format)

    xaxis_lowlimit = datetime.now(pytz.timezone('US/Pacific'))
    xaxis_uplimit = datetime.now(pytz.timezone('US/Pacific')) + timedelta(days=9)
    ax1.set_xlim([xaxis_lowlimit, xaxis_uplimit])

    ax1.set_ylabel('Snow Level (ft)', color=color)
    #ax1.set_xlabel('Date')
    ax1.set_ylim([0.0, 10000])
    #ax1.yaxis.grid(True)

    # --Start-- Create a color gradient under the curve
    alpha = 1.0
    z = np.empty((100, 1, 4), dtype=float)
    rgb = mcolors.colorConverter.to_rgb(color)
    z[:, :, :3] = rgb
    z[:, :, -1] = np.linspace(0, alpha, 100)[:, None]

    snow_series = df[f'snowLevel_{model}'].dropna()
    if not snow_series.empty and snow_series.shape[0] > 1:
        xvals = mdates.date2num(snow_series.index.values)
        yvals = snow_series.values
        xmin, xmax, ymin, ymax = xvals.min(), xvals.max(), 0, 10000

        im = ax1.imshow(z, aspect='auto', extent=[xmin, xmax, ymin, ymax],
                       origin='lower', zorder=1)

        xy = np.column_stack([xvals, yvals])
        xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
        clip_path = Polygon(xy, facecolor='none', edgecolor='none', closed=True)
        ax1.add_patch(clip_path)
        im.set_clip_path(clip_path)
        line = ax1.plot(xvals, yvals, color=[0, 57/255, 148/255], label="Snow Level")
    else:
        line = ax1.plot([], [], color=[0, 57/255, 148/255], label="Snow Level")
    # --End-- Color Gradient

    ax2 = ax1.twinx()
    ax2.set_ylim([0.0, 3.0])
    ax2.xaxis.set_major_formatter(date_format)
    color = 'tab:green'
    ax2.set_ylabel('24-hour Precip Forecast (inches)', color=color)
    qpf_bar = ax2.bar(daily_df.index, daily_df[f'qpf_hr_{model}'], color=color, alpha=0.7,
            label="Total Precip")
    sn_qpf_bar = ax2.bar(daily_df.index, daily_df[f'snow_hr_{model}'], color='tab:blue', alpha=0.7,
            label="Precip with Snow Level <= 5000 ft")

    rects = ax2.patches
    # Make some labels.
    (y_bottom, y_top) = ax2.get_ylim()
    y_height = y_top - y_bottom

    for rect_cnt, rect in enumerate(rects):
        height = rect.get_height()
        label = height
        # If a precip value is off the chart, put the label at the top of the bar so the value is known.
        if height > y_top:
            height = y_top - 0.06

        label_position = height + (y_height * 0.01)

        # Since the 06Z run will include forecast times valid yesterday, we want to remove those from the graph
        # if we don't include this if statement first, it will plot values along the -side of the x axis (in the
        # margins).
        if rect.get_x() > mdates.date2num(xaxis_lowlimit):
            # First group of bars (qpf)
            if len(rects)/2 > rect_cnt:
                ax2.text(rect.get_x() + rect.get_width() / 2., label_position,
                        f'{str(round(label,2))}',
                        ha='center', va='bottom', color='darkgreen')

            # Second group of bars (frozen qpf)
            else:
                label_position = height - (y_height * 0.07)
                if height > 0.3:
                    ax2.text(rect.get_x() + rect.get_width() / 2., label_position,
                             f'{str(round(label, 2))}',
                             ha='center', va='bottom', color='blue')


    ax1.set_zorder(ax2.get_zorder() + 1)  # put ax1 in front of ax2
    #ax1.xaxis.set_major_locator(ticker.MultipleLocator(xaxis_tick_spacing))
    ax1.patch.set_visible(False)  # hide the 'canvas'
    fig.autofmt_xdate()
    ax1.tick_params(axis='x', rotation=30)

    # Get labels for the legend from both axis, then display legend at below the dates.
    handles, labels = [], []
    for ax in fig.axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.append(h)
            labels.append(l)

    # Reorder the labels in legend as Total Precip, Snow, Snow Level.
    myorder = [1,2,0]
    handles = [handles[i] for i in myorder]
    labels = [labels[i] for i in myorder]

    # Give the bottom of the plot some extra room for the legend
    fig.subplots_adjust(bottom=0.25)

    # Add the legend to the plot
    plt.legend(handles=handles, labels=labels, bbox_to_anchor=(1, 0), loc="lower right",
               bbox_transform=fig.transFigure, frameon=False, ncol=3)
    plt.savefig(os.path.join(imgdir, 'qpf_graph.png'))

    #gradient_bar(df, xaxis_lowlimit, xaxis_uplimit)
    # plt.show()
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
