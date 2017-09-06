''' sunpan excise for simple ITK'''
a = [1, 2, 3, 4, 5]
a[-1]    # last item in the array
a[-2:]   # last two items in the array
a[:-2]   # everything except the last two items

import SimpleITK as sitk
from downloaddata import fetch_data, fetch_data_all
print(sitk.Version())
fetch_data_all(os.path.join('..','Data'), os.path.join('..','Data','manifest.json'))

from __future__ import print_function
image = sitk.Image(256, 128, 64, sitk.sitkInt16)
image_2D = sitk.Image(64, 64, sitk.sitkFloat32)
image_2D = sitk.Image([32,32], sitk.sitkUInt32)
image_RGB = sitk.Image([128,128], sitk.sitkVectorUInt8, 3)
help(image)
help(image.GetDepth)
print(image.GetSize())
print(image.GetOrigin())
print(image.GetSpacing())
print(image.GetDirection())
print(image.GetNumberOfComponentsPerPixel())
print(image.GetWidth())
print(image.GetHeight())
print(image.GetDepth())
print(image.GetDimension())
print(image.GetPixelIDValue())
print(image.GetPixelIDTypeAsString())
print(image.GetPixel(0, 0, 0))
image.SetPixel(0, 0, 0, 1)
print(image.GetPixel(0, 0, 0))
nda = sitk.GetArrayFromImage(image)# Get a view of the image data as a numpy array, useful for display
img = sitk.GetImageFromArray(nda)
img = sitk.GetImageFromArray(nda, isVector=True)
# ITK's Image class does not have a bracket operator. It has a GetPixel which takes an ITK Index object as an argument, which is an array ordered as (x,y,z). This is the convention that SimpleITK's Image class uses for the GetPixel method as well.
# While in numpy, an array is indexed in the opposite order (z,y,x).
print(img.GetSize())
print(nda.shape)
print(nda.shape[::-1])
sitk.Show(image)
z = 0
slice = sitk.GetArrayViewFromImage(image)[z,:,:]
plt.imshow(slice)

img = sitk.GaussianSource(size=[64]*2)
plt.imshow(sitk.GetArrayViewFromImage(img))
img = sitk.GaborSource(size=[64]*2, frequency=.03)
plt.imshow(sitk.GetArrayViewFromImage(img))
def myshow(img):
    nda = sitk.GetArrayViewFromImage(img)
    plt.imshow(nda)
myshow(img)
img[24,24]
myshow(img[16:48,:]) # cropping image
myshow(img[:,16:-16])
myshow(img[:32,:32])
img_corner = img[:32,:32]
myshow(img_corner)
myshow(img_corner[::-1,:])
myshow(sitk.Tile(img_corner, img_corner[::-1,::],img_corner[::,::-1],img_corner[::-1,::-1], [2,2]))

from downloaddata import fetch_data as fdata
img = sitk.ReadImage(fdata("cthead1.png"))
img = sitk.Cast(img,sitk.sitkFloat32)
myshow(img)
img[150,150]
timg = img**2
myshow(timg)
timg[150,150]