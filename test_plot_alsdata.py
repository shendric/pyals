# -*- coding: utf-8 -*-

from plib.aircraft.als import AirborneLaserScannerFile, AlsDEM, AlsDemMap
from plib.helpers.config import configuration_file


def test_plot_alsdata():
    """
    Reads a subsection of an airborne laserscanner file,
    calculates a regular gridded DEM and creates a map plot
    """
    # Get input information from configuration file
    als_example = configuration_file(r"als_example.json")
    config = als_example["test_dataset"]

    # Read the file
    als = AirborneLaserScannerFile()
    als.connect(config.file.path)
    als.set_time_range(config.file.range)
    als.read_data()

    # Create the DEM
    dem = AlsDEM(**config.dem.griddata)
    dem.set_data(als)
    dem.set_resolution(config.dem.resolution)
    dem.set_align_heading(config.dem.align_heading)
    dem.set_maximum_filter(config.dem.use_maximum_filter,  **config.dem.maximum_filter)
    dem.griddata()

    # Plot the DEM
    fig = AlsDemMap()
    fig.set_dem(dem)
    fig.set_hillshade(config.map.use_hillshade, **config.map.hillshade)
    fig.set_cmap(**config.map.colormap)
    fig.set_label(**config.map.label)
    fig.quickview()

if __name__ == '__main__':
    test_plot_alsdata()
