import rasterio
from rasterio.enums import Resampling

# Open the input file
with rasterio.open("Preprocessing\\data\\UOG_2057\\orthos\\export-data\\orthomosaic_visible.tif") as src:
    # Get the blue band (assuming it's the 3rd band)
    blue = src.read(2)

    # Calculate the new shape
    new_height = blue.shape[0] // 5
    new_width = blue.shape[1] // 5

    # Resample data to target shape
    blue_downscaled = src.read(
        2,
        out_shape=(new_height, new_width),
        resampling=Resampling.lanczos
    )

    # Update the transform
    transform = src.transform * src.transform.scale(
        (src.width / new_width),
        (src.height / new_height)
    )

    # Copy the metadata
    profile = src.profile

    # Update the metadata
    profile.update(
        driver='GTiff',
        height=new_height,
        width=new_width,
        transform=transform,
        count=1  # we're only writing one band
    )

# Write the output file
with rasterio.open("Blue_scaled.tif", 'w', **profile) as dst:
    dst.write(blue_downscaled, 1)

print("Image processing complete. Output saved as 'output_image.tif'")