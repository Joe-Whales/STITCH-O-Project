import argparse
import math
import itertools
import numpy as np
import os
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

import re
import shutil
import sys
import tqdm
import tqdm.contrib
import tqdm.contrib.itertools
import yaml

def getTIFDimensions(file_name):
    """
    RETURNS WIDTH AND HEIGHT FOR TIF FILE
    """
    with rasterio.open(file_name) as file:
        return file.width, file.height

def get_patch_size(temp_dir, patch_size_deg, overlap_size_deg):
    """
    given a patch size in degrees, calculate the number of pixels for the patch size (degrees to number of pixels x and y)
    """
    # all files have been clipped to the same extent and scaled to the same resolution and therefore you can read pixel size from any of the files
    files = os.listdir(temp_dir)
    reference = files[0]

    with rasterio.open(os.path.join(temp_dir, reference)) as src:
        # get the pixel size in degrees
        pixel_size_x = src.transform.a
        pixel_size_y = src.transform.e

        e = 10 ** -6

        # calculate the number of pixels for the patch size
        patch_size_x = int(patch_size_deg / pixel_size_x) * e
        patch_size_y = int(patch_size_deg / pixel_size_y) * e
        overlap_size_x = int(overlap_size_deg / pixel_size_x) * e
        overlap_size_y = int(overlap_size_deg / pixel_size_y) * e

        return int(patch_size_x), abs(int(patch_size_y)), int(overlap_size_x), abs(int(overlap_size_y))

def clip_then_upscale(files, upscale_width, upscale_height, output_dir, reference, verbose=False):
    """
    CLIP RASTERS WITH RESPECT TO THE EXTENT WITH RESPECT TO ... AND THEN UPSCALE ALL RASTERS TO THE SAME SIZE AND WRITE TO TEMP
    """
    os.makedirs(output_dir, exist_ok=True)
    for file_name in tqdm.tqdm(files, desc="Upscaling files", disable=verbose):
        local_name = file_name.split("/")[-1]
        if verbose:
            print(os.path.join(output_dir, local_name))
        if (not os.path.exists(os.path.join(output_dir, local_name))):
            # open raster
            with rasterio.open(file_name) as src:
                # open raster you're going reference to clip and load bounds
                with rasterio.open(reference) as target:
                    target_bounds = target.bounds
                
                # create window matching bounds
                window = from_bounds(
                    target_bounds.left, target_bounds.bottom, 
                    target_bounds.right, target_bounds.top, 
                    transform=src.transform
                )
                
                # read from file within the window
                data = src.read(window=window)
                window_transform = src.window_transform(window)
                
                # update metadata
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": data.shape[1],
                    "width": data.shape[2],
                    "transform": window_transform,
                    "crs": src.crs
                })

                # write out clipped raster
                temp_path = os.path.join(output_dir, "temp_clipped_raster.tif")
                with rasterio.open(temp_path, "w", **out_meta) as dest:
                    dest.write(data)


                    src.close()
                    dest.close()
            
            
                    src.close()
                    dest.close()
            
            # read newly written clipped raster
            with rasterio.open(temp_path) as src:
                nodata = src.nodata
                
                color_interps = src.colorinterp

                # upscale to match width and height
                data = src.read(
                    out_shape=(
                        src.count,
                        upscale_height,
                        upscale_width
                    ),
                    resampling=Resampling.bilinear
                )

                # adjust metadata
                transform = src.transform * src.transform.scale(
                    (src.width / data.shape[-1]),
                    (src.height / data.shape[-2])
                )

                output_path = os.path.join(output_dir, os.path.basename(file_name))
                with rasterio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    height=upscale_height,
                    width=upscale_width,
                    count=src.count,
                    dtype=data.dtype,
                    crs=src.crs,
                    transform=transform,
                    nodata=nodata
                ) as dst:
                    dst.write(data)
                    dst.colorinterp = color_interps
        else:
            if verbose:
                print(f"{file_name} ALREADY EXISTS IN FOLDER")
        
def create_blocks_upscaling(orchard_id, files, dimensions, upscale_width, upscale_height, block_size, overlap, anomaly_threshold, output_dir, mask_file="mask.tif", reference="reg.tif",  verbose=False):
    """
    CREATES BLOCK BIN FILES FOR VARIOUS DIMENSIONS
    PARAMETERS: Root path to files(for block naming), List of file names, list of bands for each file, desired width and height to upscale to, block size, block overlap, output directory for chunks
    """
    # make directories
    case_1_dir = os.path.join(output_dir, "case_1")
    case_2_dir = os.path.join(output_dir, "case_2")
    case_3_dir = os.path.join(output_dir, "case_3")
    normal_dir = os.path.join(output_dir, "normal")
    os.makedirs(case_1_dir, exist_ok=True)
    os.makedirs(case_2_dir, exist_ok=True)
    os.makedirs(case_3_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)
    # upscale all tif images
    temp_dir = os.path.join(output_dir, "temp/")
    files.append(mask_file), files.append(reference)
    files = list(map(lambda f: f.replace('\\', '/'), files))
    clip_then_upscale(files, upscale_width, upscale_height, temp_dir, reference=reference, verbose=verbose)

    block_size_x, block_size_y, overlap_x, overlap_y = get_patch_size(temp_dir, block_size, overlap)
    print (f"Block size: {block_size_x}x{block_size_y} px, Overlap: {overlap_x}x{overlap_y} px")

    # open mask layer
    mask = rasterio.open(os.path.join(temp_dir, os.path.basename(mask_file)))

    # open the nodata reference file
    nodata_ref = rasterio.open(os.path.join(temp_dir, os.path.basename(reference)))

    for file_name, num_bands in zip(files, dimensions):
        local_name = file_name.split("/")[-1]
        with rasterio.open(os.path.join(temp_dir, local_name)) as src:
            # block starting positions
            y_starts = np.arange(0, upscale_height - block_size_y + 1, block_size_y - overlap_y)
            x_starts = np.arange(0, upscale_width - block_size_x + 1, block_size_x - overlap_x)

            # determines whether to show progress bar or not. Don't show if verbose is enabled since the print statements get in the way of the progress bar
            if verbose:
                loop_product = itertools.product(enumerate(y_starts), enumerate(x_starts))
            else:
                loop_product = tqdm.contrib.itertools.product(enumerate(y_starts), enumerate(x_starts), total=len(y_starts) * len(x_starts), desc=f"Processing {local_name}")
            
            for (i, y), (j, x) in loop_product:
                # read in block from upscaled image
                win = Window(x, y, block_size_x, block_size_y)
                # read in block from mask
                mask_block = mask.read(1, window=win)
                nodata_ref_block = nodata_ref.read(1, window=win)
                    
                # ignore these regions
                if (not np.any(mask_block == 0)) and (not np.any(nodata_ref_block == -32767)):
                    # stack each band data to block array
                    block = np.stack([src.read(b, window=win) for b in range(1, num_bands + 1)])
                    
                    case_1_count = np.count_nonzero(mask_block == 1)
                    case_2_count = np.count_nonzero(mask_block == 2)
                    case_3_count = np.count_nonzero(mask_block == 3)
                    
                    # check for anomaly cases
                    if case_1_count >= anomaly_threshold * block_size_x * block_size_y:
                        block_file_name = os.path.join(case_1_dir, f"{orchard_id}_block_{i}_{j}.npy")
                    elif case_2_count >= anomaly_threshold * block_size_x * block_size_y:
                        block_file_name = os.path.join(case_2_dir, f"{orchard_id}_block_{i}_{j}.npy")
                    elif case_3_count >= anomaly_threshold * block_size_x * block_size_y:
                        block_file_name = os.path.join(case_3_dir, f"{orchard_id}_block_{i}_{j}.npy")
                    elif case_1_count + case_2_count + case_2_count == 0:
                        block_file_name = os.path.join(normal_dir, f"{orchard_id}_block_{i}_{j}.npy")
                    else:
                        continue

                    # check if block file exists and if so then append to data and write
                    if os.path.exists(block_file_name):
                        temp_block_data = np.load(block_file_name)
                        temp_block_data = temp_block_data.transpose(2, 0, 1)
                        updated_block = np.concatenate((temp_block_data, block), axis=0)
                    else:
                        # initialize file if block doesn't exist
                        updated_block = block
                            
                    updated_block = updated_block.transpose(1, 2, 0)
                    np.save(block_file_name, updated_block)
                    if verbose:
                        print(f'SAVED BLOCK {i}, {j} TO {block_file_name}')
                else:
                    if verbose:
                        print(f'SKIPPING BLOCK {i}, {j}')
    
    # delete temp upscale tif folder and contents
    mask.close()
    nodata_ref.close()
    src.close()
    shutil.rmtree(temp_dir)

def test_upscaling(files, upscale_width, upscale_height, output_dir):
    """
    TEST UPSCALING OF TIF IMAGES
    """
    os.makedirs(output_dir, exist_ok=True)

    for file_name in files:
        with rasterio.open(file_name) as src:
            # print(f"UPSCALING {file_name} WITH DIMENSIONS: WIDTH {src.width}, HEIGHT {src.height} TO: WIDTH {upscale_width}, HEIGHT {upscale_height}")

            nodata = src.nodata
            color_interps = src.colorinterp

            data = src.read(
                out_shape=(
                    src.count,
                    upscale_height,
                    upscale_width
                ),
                resampling=Resampling.bilinear
            )

            # adjust metadata
            transform = src.transform * src.transform.scale(
                (src.width / data.shape[-1]),
                (src.height / data.shape[-2])
            )

            output_path = os.path.join(output_dir, os.path.basename(file_name))
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=upscale_height,
                width=upscale_width,
                count=src.count,
                dtype=data.dtype,
                crs=src.crs,
                transform=transform,
                nodata=nodata
            ) as dst:
                dst.write(data)
                dst.colorinterp = color_interps

def get_files_from_project(project_dir: str) -> list[str]:
    return [project_dir + '/orthos/' + f for f in [
        'data-analysis/lwir.tif',
        'data-analysis/red.tif',
        'data-analysis/reg.tif',
        'export-data/orthomosaic_visible.tif'
    ]]

def main():
    parser = argparse.ArgumentParser(description="A tool for chunking large orthomosaic TIF files into smaller patches.")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable verbose output")
    parser.add_argument("config", type=str, help="path to the .yaml configuration file")

    args = parser.parse_args()

    config_path = args.config
    verbose = args.verbose

    if not os.path.exists(config_path):
        print(f"Config file {config_path} does not exist.")
        print("Usage: python chunker.py [config_file_path]")
        exit()


    with open(config_path, "r") as f:
        config_documents = yaml.safe_load_all(f)
        for _, config in enumerate(config_documents):
            # PARAMETERS
            try:
                path = config["path"]
                files = []
                dimensions = []
                for file in config["files"]:
                    files.append(os.path.join(path, file["name"]))
                    dimensions.append(file["dimensions"])
                chunk_size = config["chunk_size"]
                chunk_overlap = config["chunk_overlap"]
                # this represents the percentage overlap a chunk needs to have with an anomalous region to be considered anomalous
                #normal_threshold = config["normal_threshold"]
                anomaly_threshold = config["anomaly_threshold"]
                
                scale_ratio = config["scale_ratio"]
                
                # get width and height of RGB file for orchard and times each by scale factor (new height and width of each channel for this particular orchard)
                rgb_path = os.path.join(path, "orthos", "export-data", "orthomosaic_visible.tif")
                width, height = getTIFDimensions(rgb_path)

                scale_width = int(scale_ratio * width)
                scale_height = int(scale_ratio * height)

                if verbose:
                    print(f"Scaling every channel to: {scale_width}x{scale_height}")
                
                # mask used to hide outer boundary and unwanted regions (buildings, etc.). Also used to determine which blocks are normal, case 1, or case 2 anomalies
                mask_file = config["mask"]
                # reference file used to determine nodata regions and clip the rasters against before upscaling
                reference_file = config["reference"]
                output_dir = config["output_path"]

            except KeyError as e:
                field = re.findall(r"'(.+?)'", str(e))[-1]
                print(f"Missing required field '{field}' in .yaml configuration document.")
                exit()

            print(f"Chunking project: {path}")
            orchard_id = path.split(os.path.sep)[-1]
            create_blocks_upscaling(orchard_id, files, dimensions, scale_width, scale_height, chunk_size, chunk_overlap, anomaly_threshold, output_dir, mask_file, reference_file, verbose=verbose)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Cancelling job...")
        exit()