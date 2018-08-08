from PIL import Image

# ------------- .png to pixels ----------------------------------------------------------------------------------
#
# convert a .png image to a list of pixel values
#   mode = P (8-bit pixels, mapped to any other mode using a color palette)
#
# ------------- .png to pixels ----------------------------------------------------------------------------------
print('''
.png to pixels

      mode = P (8-bit pixels, mapped to any other mode using a color palette)

''')
img_name = 'example5.png'
img = Image.open(img_name).convert('P')
WIDTH, HEIGHT = img.size
data = list(img.getdata()) # convert image data to a list of integers
print('''
img_name: {}
WIDTH: {}
HEIGHT: {}
img.mode: {}
img.size: {}
image.data: {}

'''.format(img_name, WIDTH, HEIGHT, img.mode, img.size, data))

# convert image data list of integer pixel values to 2D list that can be displayed as an image
data = [data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]

# At this point the image's pixels are all in memory and can be accessed
# individually using data[row][col].

print('Display the 2D list of pixel values as an image')
for row in data:
    print(' '.join('{:3}'.format(value) for value in row))


# ------------- pixels to .png ----------------------------------------------------------------------------------
#
# convert a list of pixel values to a .png image
#   mode = P (8-bit pixels, mapped to any other mode using a color palette)
#
# ------------- pixels to .png ----------------------------------------------------------------------------------
print('''



pixels to .png
        mode = P (8-bit pixels, mapped to any other mode using a color palette)

''')
image_name = '0.png'
# mnist standard image format is 28x28
WIDTH = 28
HEIGHT = 28
# image data as a list of integers
json_data = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,92,216,234,211,37,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,80,213,251,253,253,253,217,35,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,41,151,253,253,253,253,253,253,253,145,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,13,146,231,253,253,236,162,82,189,243,253,253,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,74,187,253,250,224,103,53,0,0,0,179,253,253,0,0,0,0,0,0,0,0,0,0,0,0,0,0,68,250,253,253,134,0,0,0,0,0,0,179,253,253,0,0,0,0,0,0,0,0,0,0,0,24,111,192,250,253,214,31,2,0,0,0,0,0,61,252,253,253,0,0,0,0,0,0,0,0,0,0,0,104,253,253,253,107,17,0,0,0,0,0,0,0,63,253,253,130,0,0,0,0,0,0,0,0,0,0,100,242,253,212,114,2,0,0,0,0,0,0,0,0,174,253,236,22,0,0,0,0,0,0,0,0,0,98,242,253,202,28,0,0,0,0,0,0,0,0,0,87,239,253,206,0,0,0,0,0,0,0,0,0,77,246,253,202,29,0,0,0,0,0,0,0,0,0,12,224,253,249,85,0,0,0,0,0,0,0,0,0,195,253,210,27,0,0,0,0,0,0,0,0,0,71,189,253,251,131,0,0,0,0,0,0,0,0,0,0,254,253,111,0,0,0,0,0,0,0,0,0,4,187,253,253,191,0,0,0,0,0,0,0,0,0,0,0,254,253,40,0,0,0,0,0,0,0,0,83,241,253,253,119,10,0,0,0,0,0,0,0,0,0,0,0,210,253,110,20,0,0,0,0,0,68,183,242,253,253,174,23,0,0,0,0,0,0,0,0,0,0,0,0,89,248,253,182,63,63,63,196,199,243,253,253,251,125,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,122,212,243,253,253,253,253,253,253,223,197,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,73,188,232,232,232,232,121,26,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]


# mode = P (8-bit pixels, mapped to any other mode using a color palette)
image_mode = 'P'
image_size = (WIDTH, HEIGHT)
print('''
image_name: {}
WIDTH: {}
HEIGHT: {}
img.mode: {}
img.size: {}
image.data: {}

'''.format(image_name, WIDTH, HEIGHT, image_mode, image_size, json_data))

image_out = Image.new(image_mode, image_size)
image_out.putdata(json_data)
image_out.save(image_name)

json_data = [json_data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]

print('Display the 2D list of pixel values as an image')
for row in json_data:
    print(' '.join('{:3}'.format(value) for value in row))
