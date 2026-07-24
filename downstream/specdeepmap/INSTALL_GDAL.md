# Installing GDAL on Windows

GDAL (Geospatial Data Abstraction Library) is required for reading geospatial raster files. Here are installation options:

## Option 1: Using Conda (Recommended - Easiest)

If you're using conda (which you are, based on your environment `specdeepmap_august_2025`):

```bash
conda install -c conda-forge gdal
```

This is the easiest method and handles all dependencies automatically.

## Option 2: Using pip (More Complex)

If you prefer pip, you need to install GDAL system libraries first, then Python bindings:

1. **Download GDAL binaries for Windows:**
   - Go to https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal
   - Download the appropriate `.whl` file for your Python version and architecture
   - Or use: `pip install GDAL` (may not work on all systems)

2. **Set environment variables** (if needed):
   ```bash
   set GDAL_DATA=C:\path\to\gdal\data
   set GDAL_DRIVER_PATH=C:\path\to\gdal\plugins
   ```

## Option 3: Using OSGeo4W (For QGIS Users)

If you have QGIS installed, GDAL might already be available. You may need to:
- Add QGIS Python to your PATH
- Or copy GDAL DLLs to your conda environment

## Verify Installation

After installing, verify GDAL works:

```python
from osgeo import gdal
print(gdal.__version__)
```

## Troubleshooting

If you still get `No module named '_gdal'`:

1. **Check if GDAL is in your environment:**
   ```bash
   conda list gdal
   ```

2. **Try reinstalling:**
   ```bash
   conda remove gdal
   conda install -c conda-forge gdal
   ```

3. **Check Python version compatibility:**
   - Make sure GDAL version matches your Python version
   - For Python 3.9-3.11, use: `conda install -c conda-forge gdal=3.7`

4. **Restart your terminal** after installation

## Quick Fix for Your Current Issue

Run this in your conda environment:

```bash
conda install -c conda-forge gdal
```

Then try running `train.py` again.
