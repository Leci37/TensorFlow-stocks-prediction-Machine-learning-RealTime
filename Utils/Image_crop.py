#https://stackoverflow.com/questions/10615901/trim-whitespace-using-pil

from PIL import Image, ImageChops
def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    else:
        # Failed to find the borders, convert to "RGB"
        return trim(im.convert('RGB'))

im = Image.open(r"C:\Users\Luis\Downloads\MELI_YahooFinanceChart (2).png")
im = trim(im)
im.save('bee2.png', optimize=True, format="PNG")
im.show()