import openslide
import matplotlib.pyplot as plt
%pylab
inline

# Construct slide object.
slide = openslide.OpenSlide('CMU-1.tiff')
# Output its parameters.
print(slide.level_count)
print(slide.dimensions)
print(slide.level_dimensions)
print(slide.level_downsamples)
print(slide.properties)

# Output thumbnail and display.
thumbnail = slide.get_thumbnail((1000, 1000))
plt.imshow(thumbnail)

# Detects the format of the file.
print(openslide.OpenSlide.detect_format('CMU-1.tiff'))
# Read layer 1, coordinates at (11000, 14000), size 500x500 image, and display.
region = slide.read_region((11000, 14000), 1, (500, 500))
plt.imshow(region)
