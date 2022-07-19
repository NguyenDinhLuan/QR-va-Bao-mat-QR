
from pyzbar.pyzbar import decode
from PIL import Image


img= Image.open('QRcode.png')
output=decode(img)

print("Du lieu ma QR sau khi giai ma:")
print(output)