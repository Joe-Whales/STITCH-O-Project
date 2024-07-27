import rasterio
from rasterio.enums import Resampling
import numpy as np
from multiprocessing import Pool, cpu_count
from PIL import Image

def process_chunk(args):
    src_file, window, scale_factor = args
    with rasterio.open(src_file) as src:
        blue = src.read(3, window=window)
        blue_downscaled = np.array(Image.fromarray(blue).resize(
            (blue.shape[1] // scale_factor, blue.shape[0] // scale_factor),
            resample=Image.LANCZOS
        ))
    return blue_downscaled

def main():
    input_file = "Preprocessing\\data\\UOG_2057\\orthos\\export-data\\orthomosaic_visible.tif"
    output_file = "output_image.tif"
    scale_factor = 5
    chunk_size = 5000  # Adjust this based on your available memory

    with rasterio.open(input_file) as src:
        profile = src.profile.copy()
        new_width = src.width // scale_factor
        new_height = src.height // scale_factor

        profile.update(
            driver='GTiff',
            height=new_height,
            width=new_width,
            count=1,
            transform=src.transform * src.transform.scale(
                (src.width / new_width),
                (src.height / new_height)
            )
        )

        # Prepare chunks
        chunks = []
        for y in range(0, src.height, chunk_size):
            for x in range(0, src.width, chunk_size):
                window = rasterio.windows.Window(x, y, 
                                                 min(chunk_size, src.width - x), 
                                                 min(chunk_size, src.height - y))
                chunks.append((input_file, window, scale_factor))

    # Process chunks in parallel
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_chunk, chunks)

    # Write results
    with rasterio.open(output_file, 'w', **profile) as dst:
        for i, result in enumerate(results):
            window = rasterio.windows.Window(
                i % (src.width // chunk_size) * (chunk_size // scale_factor),
                i // (src.width // chunk_size) * (chunk_size // scale_factor),
                result.shape[1],
                result.shape[0]
            )
            dst.write(result, 1, window=window)

    print("Image processing complete. Output saved as 'output_image.tif'")

if __name__ == '__main__':
    main()