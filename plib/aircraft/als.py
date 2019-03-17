# User Libraries
from plib.helpers.objects import EmptyObject
from plib.mpl.shading import set_shade, shadow_from_alpha
from plib.mpl.vacumm_color import cmap_previmer2

# Projection
from pyproj import Proj
# import utm

# Numpy & Scipy
import numpy as np
from scipy.ndimage.filters import maximum_filter

# Matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# from matplotlib.mlab import griddata
from scipy.interpolate import griddata

# All other stuff
from collections import OrderedDict
import struct
import warnings
import logging

# TODO: use logging
# TODO: use configuration file for parsing of ALS data

# Default settings


class ALSL1BFileDefinition():
    """ TODO: Move this to configuration file """
    def __init__(self):
        self.set_header_info()
        self.set_line_variables()

    def set_line_variables(self):
        self.line_variables = OrderedDict((('timestamp', np.float64),
                                           ('longitude', np.float64),
                                           ('latitude', np.float64),
                                           ('elevation', np.float64),
                                           ('amplitude', np.float32),
                                           ('reflectance', np.float32)))

    def set_header_info(self):
        self.header_dict = OrderedDict((('scan_lines', [4, '>L']),
                                        ('data_points_per_line', [2, '!H']),
                                        ('bytes_per_line', [2, '>H']),
                                        ('bytes_sec_line', [8, '>Q']),
                                        ('year', [2, '>H']),
                                        ('month', [1, '>b']),
                                        ('day', [1, '>b']),
                                        ('start_time_sec', [4, '>L']),
                                        ('stop_time_sec', [4, '>L']),
                                        ('device_name', [8, '>8s'])))


class AirborneLaserScannerFile(object):
    """ Not more than a proof of concept yet """
    def __init__(self):
        self.filename = None
        self.header = EmptyObject()
        self.line_timestamp = None
        self.connected = False
        self.n_selected_lines = 0
        self.filedef = ALSL1BFileDefinition()

    def connect(self, filename):
        """
        Connect to ALS Level1b binary file: retrieve header information
        """
        self.filename = filename
        self.read_header()
        self.read_line_timestamp()
        self.connected = True
        self.set_full_time_range()

    def read_header(self, verbose=True):
        """ Read the header of the ALS level 1b file """
        header_dict = self.filedef.header_dict
        with open(self.filename, 'rb') as f:
            # Read header size
            self.header.byte_size = struct.unpack('>b', f.read(1))[0]
            logging.info("als_header.byte_size: %s" %
                         str(self.header.byte_size))
            if self.header.byte_size == 36:
                header_dict['data_points_per_line'] = [1, '>B']
            elif self.header.byte_size == 37:
                header_dict['data_points_per_line'] = [2, '>H']
            else:
                raise ValueError("Unkown ALS L1B header size:",
                                 self.header.byte_size,
                                 "\nShould be 36 or 37 or unsupported Device")
            # Read Rest of header
            for key in header_dict.keys():
                nbytes, fmt = header_dict[key][0], header_dict[key][1]
                setattr(self.header, key,
                        struct.unpack(fmt, f.read(nbytes))[0])
                if verbose:
                    logging.info("als_header.%s: %s" %
                                 (key, str(getattr(self.header, key))))

    def read_line_timestamp(self):
        """ Read the line time stamp """
        with open(self.filename, 'rb') as f:
            f.seek(self.get_start_byte_linetimestamp())
            data = f.read(self.header.bytes_sec_line)
        struct_def = ">{scan_lines}L".format(scan_lines=self.header.scan_lines)
        self.line_timestamp = np.array(struct.unpack(struct_def, data))

    def read_data(self):
        """  Read lines defined by start and stop second of day """
        self.validate_file()
        startbyte, nbytes = self.get_data_bytes()
#        bindat_dtype = "s%s" % str(nbytes)
        bindat = np.ndarray(shape=(self.n_selected_lines), dtype=object)
        with open(self.filename, 'rb') as f:
            for i in np.arange(self.n_selected_lines):
                f.seek(startbyte)
                bindat[i] = f.read(nbytes)
                startbyte += nbytes
            # bindat = np.fromfile(f, count=nbytes)
        self.set_data_variables()
        self.data = self.unpack_binary_line_data(bindat)

    def unpack_binary_line_data(self, bindat):
        """ TODO: working solution, but needs to be improved """
        nlines = self.n_selected_lines
        nshots = self.header.data_points_per_line
        start_byte, stop_byte = 0, self.header.bytes_per_line
        for i in np.arange(nlines):
            line = bindat[i]
            i0, i1 = 0, 8*nshots
            self.timestamp[i, :] = struct.unpack(
                ">{n}d".format(n=nshots), line[i0:i1])
            i0 = i1
            i1 = i0 + 8*nshots
            self.latitude[i, :] = struct.unpack(
                ">{n}d".format(n=nshots), line[i0:i1])
            i0 = i1
            i1 = i0 + 8*nshots
            self.longitude[i, :] = struct.unpack(
                ">{n}d".format(n=nshots), line[i0:i1])
            i0 = i1
            i1 = i0 + 8*nshots
            start_byte += self.header.bytes_per_line
            stop_byte += self.header.bytes_per_line
            self.elevation[i, :] = struct.unpack(">{n}d".format(n=nshots),
                                                 line[i0:i1])

    def set_data_variables(self):
        """ Create the numpy arrays for unpacking of binary line data """
        nlines = self.n_selected_lines
        nshots = self.header.data_points_per_line
        for key in self.filedef.line_variables.keys():
            setattr(self, key,
                    np.ndarray(shape=(nlines, nshots),
                               dtype=self.filedef.line_variables[key]))

    def set_time_range(self, time_range):
        """ Sets the first and last line of the subsection """
        self.validate_file()
        self.validate_time_range(time_range[0], time_range[1])
        self.line_index = [
            np.where(self.line_timestamp >= time_range[0])[0][0],
            np.where(self.line_timestamp <= time_range[1])[0][-1]]
        self.n_selected_lines = self.line_index[1] - self.line_index[0]

    def set_full_time_range(self):
        """ Set the full time range as selected content """
        self.validate_file()
        self.line_index = [0, self.header.scan_lines-1]
        self.n_selected_lines = self.header.scan_lines

    def validate_time_range(self, start, stop):
        """ Check for oddities in the time range selection """
        fstart = self.line_timestamp[0]
        fstop = self.line_timestamp[-1]
        # Raise Errors
        if start > stop:
            raise ValueError(
                "start time {start} after stop time {stop}".format(
                    start=start, stop=stop))
        if start > fstop or stop < fstart:
            raise ValueError(
                "time range {start} - {stop} out of bounds " +
                "{fstart} - {fstop}".format(
                    start=start, stop=stop, fstart=fstart, fstop=fstop))
        # Raise Warnings
        if start < fstart:
            # TODO: Use logging
            warnings.warn("start time {start} before actual start of " +
                          "file {fstart}".format(start=start, fstart=fstart))
        if stop > fstop:
            warnings.warn("stop time {stop} after actual end of file " +
                          "{fstop}".format(stop=stop, fstop=fstop))

    def validate_file(self):
        """ Check if file has been specified correctly """
        if not self.connected:
            raise IOError("not connected to file -> self.connect(filename)")

    def get_data_bytes(self):
        """ Returns start and stop bytes of the selected data section """
        # Start byte of scan line
        startbyte = np.uint32(self.header.byte_size)
        startbyte += np.uint32(self.header.bytes_sec_line)
        startbyte += np.uint32(self.line_index[0]) * \
            np.uint32(self.header.bytes_per_line)
        # Number bytes for selected scan lines
        # n_scan_lines = self.line_index[1]-self.line_index[0]
        nbytes = self.header.bytes_per_line  # * n_scan_lines
        return startbyte, nbytes

    def get_start_byte_linetimestamp(self):
        """ Returns the start byte of the line timestamp array """
        return self.header.byte_size

    def get_center_beam_index(self):
        """ Returns index of the center beam """
        if not self.connected:
            return 0
        return np.median(np.arange(self.header.data_points_per_line))

    def get_n_shots_per_line(self):
        """ Returns index of the center beam """
        if not self.connected:
            return 0
        return self.header.data_points_per_line


class AlsDEM(object):
    """ TODO: Documentation """

    MAXIMUM_FILTER_DEFAULT = {'size': 3, 'mode': 'nearest'}
    GRIDDATA_DEFAULT = {'method': 'linear', 'rescale': False}

    def __init__(self, **kwargs):
        """ TODO: Documentation """
        self._has_data = False
        self._align_heading = False
        self._use_maximum_filter = True
        self._resolution = 1.0
        self._grid_pad_fraction = 0.01
        self._dem_settings = self.GRIDDATA_DEFAULT
        self._dem_settings.update(kwargs)
        self._maximum_filter_kwargs = self.MAXIMUM_FILTER_DEFAULT

    def set_data(self, data):
        """ TODO: Documentation """
        self.longitude = data.longitude
        self.latitude = data.latitude
        self.elevation = data.elevation
        self._has_data = True

    def set_resolution(self, resolution):
        """ TODO: Documentation """
        self._resolution = resolution

    def set_maximum_filter(self, onoff, **kwargs):
        """
        TODO: this should have an automatic method to decide
        automatically what filter width to use, based on
        scanner point spacing and DEM resolution
        """
        self._use_maximum_filter = onoff
        self._maximum_filter_kwargs.update(kwargs)

    def set_align_heading(self, onoff):
        """ TODO: Documentation """
        self._align_heading = onoff

    def griddata(self, method=None, rescale=None):
        """ Grids irregular laser scanner points to regular grid """
        # TODO: Properly validate data
        self._proj()
        if self._align_heading:
            self._align()
        self._griddata()
        if self._use_maximum_filter:
            self._maximum_filter()

    def _proj(self):
        """ Calculate projection coordinates """
        # TODO: UTM will fail far up north
#        _, _, zone, letter = utm.from_latlon(np.nanmean(self.latitude),
#                                             np.nanmean(self.longitude))
        # p = Proj(proj='utm', zone=zone, ellps='WGS84')
        lat_0ts = np.nanmedian(self.latitude)
        lon_0 = np.nanmedian(self.longitude)
        p = Proj(proj='stere', lat_ts=lat_0ts, lat_0=lat_0ts, lon_0=lon_0)
        self.x, self.y = p(self.longitude, self.latitude)
        nan_mask = np.where(np.logical_or(np.isnan(self.longitude),
                                          np.isnan(self.latitude)))
        if len(nan_mask) > 0:
            self.x[nan_mask] = np.nan
            self.y[nan_mask] = np.nan

    def _griddata(self):
        """ Do the actual gridding """
        res = self._resolution
        # Get area of data
        xmin = np.nanmin(self.x)
        xmax = np.nanmax(self.x)
        ymin = np.nanmin(self.y)
        ymax = np.nanmax(self.y)
        # Add padding
        width = xmax-xmin
        height = ymax-ymin
        pad = np.amax([self._grid_pad_fraction*width,
                       self._grid_pad_fraction*height])
        xmin = np.floor(xmin - pad)
        xmax = np.ceil(xmax + pad)
        ymin = np.floor(ymin - pad)
        ymax = np.ceil(ymax + pad)
        # Create Grid and no data mask
        self.lrx = np.arange(xmin, xmax+res, res)
        self.lry = np.arange(ymin, ymax+res, res)
        self.dem_x, self.dem_y = np.meshgrid(self.lrx, self.lry)
        self.nonan = np.where(np.logical_or(np.isfinite(self.x),
                                            np.isfinite(self.y)))
        # Create regular grid
        self.dem_z = griddata((self.x[self.nonan].flatten(),
                               self.y[self.nonan].flatten()),
                              self.elevation[self.nonan].flatten(),
                              (self.dem_x, self.dem_y),
                              **self._dem_settings)
        self.dem_z = np.ma.array(self.dem_z)
        self.dem_mask = np.zeros(self.dem_z.shape, dtype=np.bool)

    def _maximum_filter(self):
        """
        Remove interpolation results in areas where no als data is available
        but which are in the concex hull of the swath
        """
        res = self._resolution
        xedges = np.linspace(self.lrx[0]-res/2.,
                             self.lrx[-1]+res/2.0, len(self.lrx)+1)
        yedges = np.linspace(self.lry[0]-res/2.,
                             self.lry[-1]+res/2.0, len(self.lry)+1)

        # Calculates point density of als shots per DEM grid cell
        self.rzhist, xe, ye = np.histogram2d(self.x[self.nonan].flatten(),
                                             self.y[self.nonan].flatten(),
                                             bins=[xedges, yedges])
        self.rzhist = self.rzhist.transpose()
        data_mask = self.rzhist > 0.0

        data_mask = maximum_filter(data_mask, **self._maximum_filter_kwargs)

#        structure = [[0,1,0],
#                     [1,1,1],
#                     [0,1,0]]
#        cluster_id, num = measurements.label(nodata_mask, structure=structure)
#        cluster_size = measurements.sum(nodata_mask, cluster_id,
#                                        index=np.arange(cluster_id.max()+1))
#        data_mask = cluster_size[cluster_id] < 50
        self.dem_mask = ~data_mask

    def _align(self):
        """
        Rotates DEM that mean flight direction
        """
        # Get angle of direction (cbi: center beam index)
        shape = np.shape(self.x)
        cbi = np.median(np.arange(len(self.x[0, :]))).astype(int)
        vec1 = [self.x[0, cbi], self.y[0, cbi],  0.0]
        vec2 = [self.x[-1, cbi], self.y[-1, cbi], 0.0]
        angle = -1.0*np.arctan((vec2[1]-vec1[1])/(vec2[0]-vec1[0]))
        # Get center point
        xc = np.nanmedian(self.x)
        yc = np.nanmedian(self.y)
        # Reform points
        points = [self.x.flatten()-xc, self.y.flatten()-yc]
        # Execute the rotation
        rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                               [np.sin(angle),  np.cos(angle)]])
        points_rotated = rot_matrix.dot(points)
        self.x = np.reshape(points_rotated[0, :], shape)
        self.y = np.reshape(points_rotated[1, :], shape)
        # Save conversion parameters for reuse
        self._align_parameters = {'center_point': (xc, yc),
                                  'angle': angle,
                                  'rotation_matrix': rot_matrix}


class AlsDemMap(object):

    DEFAULT_LABELS = {'xaxis': 'meter', 'yaxis': 'meter',
                      'zaxis': 'Elevation (meter)'}
    HILLSHADE_DEFAULT = {'scale': 0.1, 'azdeg': 165.0, 'altdeg': 45.0}
    CMAP_DEFAULTS = {'name': 'premiver2', 'range': 'percintels',
                     'qmin': 1.0, 'qmax': 99.0, 'vmin': 0.0, 'vmax': 2.0,
                     'nice_numbers': True}

    def __init__(self):
        self._has_dem = False
        self._use_hillshade = True
        self._hillshade_args = self.HILLSHADE_DEFAULT
        self._actual_heading = True
        self._cmap = cmap_previmer2()
        self._cmap_settings = self.CMAP_DEFAULTS
        self._label = self.DEFAULT_LABELS
        self.grid_spacing = 100

        self.fig = plt.figure("ALS DEM Map", figsize=(20, 10),
                              facecolor='white')
        self._fig_reference_aspect = 3

    def set_dem(self, dem):
        self.dem_x = dem.dem_x
        self.dem_y = dem.dem_y
        self.dem_z = dem.dem_z
        self.dem_mask = dem.dem_mask

    def set_cmap(self, **kwargs):
        """
        TODO: Could us some documentation
        """
        self._cmap_settings.update(kwargs)

    def set_hillshade(self, onoff, **kwargs):
        """
        TODO: Could us some documentation
        """
        self._use_hillshade = onoff
        self._hillshade_args.update(kwargs)

    def set_label(self, **kwargs):
        """
        TODO: Could us some documentation
        """
        self._label.update(kwargs)

    def set_grid_spacing(self, spacing):
        self.grid_spacing = spacing

    def quickview(self):
        """ Creates a quick and dirty DEM plot using matplotlib """

        import seaborn as sns
        sns.set_context("notebook", font_scale=1.5,
                        rc={"lines.linewidth": 1.5})
        self._plot()
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()
        plt.close(self.fig)

    def save_fig(self, filename, dpi=300):
        import seaborn as sns
        sns.set_context("talk", font_scale=1.2)
        self._plot()
        plt.savefig(filename, dpi=dpi)
        plt.clf()

    def _plot(self):

        # 2 axis: 1: DEM, 2: Colorbar
        ax1 = self.fig.add_axes([0.07, 0.30, 0.90, 0.65])
        ax2 = self.fig.add_axes([0.52, 0.15, 0.45, 0.03])
        # limits and colors
        vmin, vmax = self._get_range()
        xlim, ylim, data_extent = self._scale_axes()
        # Plot the DEM
        ax1.plot()
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        rgba = self._get_image_object()
        ax1.imshow(rgba, interpolation='none', origin='lower',
                   extent=data_extent, zorder=100)
        sigma = np.floor(len(rgba[:, 0, 0]) * 0.02)
        ax1.imshow(shadow_from_alpha(rgba, sigma=sigma), interpolation='none',
                   origin='lower', alpha=0.5,
                   extent=data_extent, zorder=99)

        ax1.xaxis.set_major_locator(ticker.MultipleLocator(self.grid_spacing))
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(self.grid_spacing))
        # ax1.set_autoscale_on(False)
        ax1.set_aspect('equal')
        ax1.xaxis.label.set_color('0.5')
        ax1.set_xlabel(self._label["xaxis"])
        ax1.yaxis.label.set_color('0.5')
        ax1.set_ylabel(self._label["yaxis"])
        ax1.tick_params(axis='both', colors='0.5')
        # Plot the colorbar
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=self._cmap,
                                        norm=norm, orientation='horizontal')
        cb1.set_label(self._label["zaxis"])

    def _scale_axes(self):
        # Get Data extent
        left = 0
        right = np.amax(self.dem_x) - np.amin(self.dem_x)
        height = np.amax(self.dem_y) - np.amin(self.dem_y)
        bottom = -0.5 * height
        top = 0.5 * height
        # Get figure extent
        figure_aspect = (right-left)/(top-bottom)
        left_off, right_off, bottom_off, top_off = 0.0, 0.0, 0.0, 0.0
        if figure_aspect > self._fig_reference_aspect:
            ref_height = (right-left) / self._fig_reference_aspect
            ref_height -= (top-bottom)
            bottom_off = -0.5*ref_height
            top_off = 0.5*ref_height
        if figure_aspect < self._fig_reference_aspect:
            ref_width = (top-bottom) * self._fig_reference_aspect
            ref_width -= (right-left)
            left_off = -0.5*ref_width
            right_off = 0.5*ref_width
        xlim = [left+left_off, right+right_off]
        ylim = [bottom+bottom_off, top+top_off]
        data_extent = [left, right, bottom, top]
        return xlim, ylim, data_extent

    def _get_percintels(self):
        """ Calculates the percintels of the elevation data """
        from plib.helpers.scaling import auto_bins
        finite = np.where(np.isfinite(self.dem_z))
        qmin = self._cmap_settings['qmin']
        qmax = self._cmap_settings['qmax']
        limits = np.percentile(self.dem_z[finite], [qmin, qmax])
        if self._cmap_settings['nice_numbers']:
            limits = auto_bins(limits[0], limits[1])
        return [np.amin(limits), np.amax(limits)]

    def _get_range(self):
        # TODO: Allow other modes (fixed range etc...)
        if self._cmap_settings['range'] == 'percintels':
            vmin, vmax = self._get_percintels()
        else:
            vmin = self._cmap_settings['vmin']
            vmax = self._cmap_settings['vmax']
        return vmin, vmax

    def _get_image_object(self):
        # TODO: Documentation
        vmin, vmax = self._get_range()
        # TODO: Allow to turn hillshade off
        dem_shaded = set_shade(self.dem_z, cmap=self._cmap,
                               vmin=vmin, vmax=vmax,
                               **self._hillshade_args)
        rgba = np.ones(self.dem_z.shape + (4,))
        rgba[:, :, 0:3] = dem_shaded
        rgba[:, :, 3] = ~self.dem_mask
        return rgba
