import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
#import geoviews as gv
#import geoviews.feature as gf
#from geoviews import dim, opts
#from geoviews.operation.regrid import regrid
#import panel as pn

class GRIB_DL():
    DEFAULT_ENDPOINT = 'nomads.ncep.noaa.gov/dods/'
    LEGACY_VARIABLE_ALIASES = {
        'snodsfc': 'Snow_depth_surface',
        'apcpsfc': 'Total_precipitation_surface_Mixed_intervals_Accumulation',
        'hgtprs': 'Geopotential_height_isobaric',
        'tmpprs': 'Temperature_isobaric',
    }
    TIME_COORD_CANDIDATES = (
        'time', 'time1', 'time2', 'time3',
        'validtime', 'validtime1', 'validtime2', 'validtime3',
    )
    LEVEL_COORD_CANDIDATES = ('lev', 'isobaric', 'isobaric1')

    def __init__(self, model=None, model_resolution='', model_run=None, date=None):
        if not model:
            model = 'gfs'
        if not model_run:
            model_run = '06z'
        if not date:
            date = datetime.utcnow().strftime("%Y%m%d")
        self.model = model
        self.model_resolution = model_resolution
        self.model_run = model_run
        self.date = date
        self.variable_aliases = {}
        self.url = self._build_url(model=model, model_resolution=model_resolution, model_run=model_run, date=date)
        self.vtimes = self.valid_times()

    def _build_url(self, model, model_resolution, model_run, date):
        # NOMADS retired OpenDAP on Feb 23, 2026. Use UCAR THREDDS OpenDAP for GFS.
        if model == 'gfs':
            run_hour = model_run.lower().replace('z', '').zfill(2)
            self.variable_aliases = self.LEGACY_VARIABLE_ALIASES.copy()
            return (
                'https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/GFS/Global_0p25deg/'
                f'GFS_Global_0p25deg_{date}_{run_hour}00.grib2'
            )

        url = f"{model}/{model}{date}/{model}_{model_resolution}_{model_run}"
        if 'nam' in model:
            url = f"{model}/{model}{date}/{model}_{model_resolution}_{model_run}"
        return f"https://{self.DEFAULT_ENDPOINT}{url}"

    def _open_dataset(self):
        try:
            return xr.open_dataset(self.url)
        except OSError as exc:
            msg = str(exc)
            if 'DAP2 DDS or DAP4 DMR' in msg or 'OpenDAP format has been retired' in msg:
                raise RuntimeError(
                    f'Failed to open OPeNDAP dataset: {self.url}. '
                    'NOMADS retired OpenDAP service on February 23, 2026 (SCN 25-81).'
                ) from exc
            raise

    def _resolve_variable_name(self, variable):
        if variable in self.variable_aliases:
            return self.variable_aliases[variable]
        return variable

    def _find_time_coord(self, data_array):
        for coord in self.TIME_COORD_CANDIDATES:
            if coord in data_array.coords:
                return coord
        return None

    def _find_level_coord(self, data_array):
        for coord in self.LEVEL_COORD_CANDIDATES:
            if coord in data_array.coords:
                return coord
        return None

    def _normalize_level_value(self, data_array, level_coord, level):
        if not isinstance(level, (int, float)):
            return level
        coord_max = float(data_array[level_coord].max())
        level_value = float(level)
        # Convert requested isobaric level from hPa (700) to Pa (70000) when needed.
        if coord_max > 2000 and level_value <= 2000:
            level_value *= 100.0
        return level_value

    def _normalize_time_dim(self, data_array):
        for coord in self.TIME_COORD_CANDIDATES:
            if coord in data_array.dims and coord != 'time':
                return data_array.rename({coord: 'time'})
        return data_array

    def valid_times(self):
        vtimes = []
        with self._open_dataset() as ds:
            time_coord = None
            for coord in self.TIME_COORD_CANDIDATES:
                if coord in ds.coords:
                    time_coord = coord
                    break
            if not time_coord:
                return vtimes
            for t in range(ds[time_coord].size):
                vtimes.append(datetime.utcfromtimestamp(ds[time_coord][t].data.astype('O') / 1e9))
        return vtimes

    def pull_point_data(self, lat, lon, variable, level, time=None):
        with self._open_dataset() as ds:
            variable_name = self._resolve_variable_name(variable)
            data_var = ds[variable_name]

            selection = {'lon': lon, 'lat': lat}
            time_coord = self._find_time_coord(data_var)
            if time and time_coord:
                selection[time_coord] = time

            if level not in ('surface', 'sfc'):
                level_coord = self._find_level_coord(data_var)
                if level_coord:
                    selection[level_coord] = self._normalize_level_value(data_var, level_coord, level)

            data = data_var.sel(**selection, method='nearest').load()
        data = self._normalize_time_dim(data)
        return data

    def pull_global_data(self, variable, level=None, time=None):
        ds = self._open_dataset()
        variable_name = self._resolve_variable_name(variable)
        data_var = ds[variable_name]
        selection = {}
        time_coord = self._find_time_coord(data_var)
        level_coord = self._find_level_coord(data_var)

        if time and time_coord:
            selection[time_coord] = time
        if level and level_coord:
            selection[level_coord] = self._normalize_level_value(data_var, level_coord, level)

        if selection:
            data = data_var.sel(**selection)
        else:
            data = data_var
        data = self._normalize_time_dim(data)
        return data

    def pull_liquid_precip_intervals(self, lat, lon, variable='apcpsfc'):
        with self._open_dataset() as ds:
            variable_name = self._resolve_variable_name(variable)
            data_var = ds[variable_name].sel(lon=lon, lat=lat, method='nearest').load()

            # UCAR GFS exposes mixed-interval accumulation with bounds. Build non-overlapping
            # interval precip by differencing the cumulative series that shares the run start.
            if 'time3' in data_var.dims and 'time3_bounds' in ds:
                bounds = ds['time3_bounds'].sel(time3=data_var['time3']).load().values
                frame = pd.DataFrame({
                    'start': pd.to_datetime(bounds[:, 0]),
                    'end': pd.to_datetime(bounds[:, 1]),
                    'accum': data_var.values.astype(float),
                })
                run_start = frame['start'].min()
                frame = frame.loc[frame['start'] == run_start].copy()
                frame.sort_values('end', inplace=True)
                frame.drop_duplicates(subset='end', keep='last', inplace=True)
                frame['interval'] = frame['accum'].diff().fillna(frame['accum']).clip(lower=0.0)

                interval = xr.DataArray(
                    frame['interval'].to_numpy(dtype=float),
                    coords={'time': frame['end'].to_numpy(dtype='datetime64[ns]')},
                    dims=('time',),
                    name=variable_name,
                )
                if 'lat' in data_var.coords:
                    interval = interval.assign_coords(lat=float(data_var['lat'].values))
                if 'lon' in data_var.coords:
                    interval = interval.assign_coords(lon=float(data_var['lon'].values))
                return interval

            normalized = self._normalize_time_dim(data_var)
            if 'time' not in normalized.dims:
                return normalized

            diffed = normalized.diff('time', label='upper')
            first = normalized.isel(time=0).expand_dims(time=[normalized['time'].values[0]])
            interval = xr.concat([first, diffed], dim='time')
            interval = interval.clip(min=0.0)
            return interval


    def plot_ds(self, ds, time, bounds=None, show=False):
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        ax = plt.axes(projection=ccrs.PlateCarree())
        if not bounds:
            bounds = [-130,-80,29,48] # West Lon, East Lon, South Lat, North Lat
        ax.set_extent(bounds)
        ax.add_feature(cfeature.STATES.with_scale('50m'))
        dsp = ds.sel(time=time)
        dsp = dsp.where(dsp.data > 0)
        dsp.plot()
        if show:
            plt.show()
        return

    def interactive_plot(self, ds, time, bounds=None, show=False):
        import cartopy.crs as ccrs

        gv.extension('bokeh')
        dsp = ds.sel(time=time)
        dsp = dsp.where(dsp.data > 0)
        refl = gv.Dataset(dsp, ['lon', 'lat', 'time'], 'refd1000m', crs=ccrs.PlateCarree())
        images = refl.to(gv.Image)
        #regridded_img = regrid(images)
        images.opts(cmap='viridis', colorbar=True, width=600, height=500) * gv.tile_sources.ESRI.options(show_bounds=True)
        #pn.panel(images).show()
        gv.save(images,'image.html')
        return

class Parameter_Builder(GRIB_DL):
    def __init__(self, models, model_resolution='', model_runs=None, dates=None):
        if not models:
            models = ['gfs']
        if not model_runs:
            model_runs = ['06z']
        if not dates:
            dates = [datetime.utcnow().strftime("%Y%m%d")]
        for model in models:
            for date in dates:
                for model_run in model_runs:
                    self.url = self._build_url(
                        model=model,
                        model_resolution=model_resolution,
                        model_run=model_run,
                        date=date,
                    )
                    self.vtimes = self.valid_times()
