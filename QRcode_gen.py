import itertools
import os.path
import numpy as np
from PIL import Image

_NUM_ERROR_CORRECTION_BLOCKS = (
    # Phien ban(0 lam phan dem, gia tri nay khong hop le)
    # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40    Error correction level
    (-1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 4,  4,  4,  4,  4,  6,  6,  6,  6,  7,  8,  8,  9,  9, 10, 12, 12, 12, 13, 14, 15, 16, 17, 18, 19, 19, 20, 21, 22, 24, 25),  # Low
    (-1, 1, 1, 1, 2, 2, 4, 4, 4, 5, 5,  5,  8,  9,  9, 10, 10, 11, 13, 14, 16, 17, 17, 18, 20, 21, 23, 25, 26, 28, 29, 31, 33, 35, 37, 38, 40, 43, 45, 47, 49),  # Medium
    (-1, 1, 1, 2, 2, 4, 4, 6, 6, 8, 8,  8, 10, 12, 16, 12, 17, 16, 18, 21, 20, 23, 23, 25, 27, 29, 34, 34, 35, 38, 40, 43, 45, 48, 51, 53, 56, 59, 62, 65, 68),  # Quartile
    (-1, 1, 1, 2, 4, 4, 4, 5, 6, 8, 8, 11, 11, 16, 16, 18, 16, 19, 21, 25, 25, 25, 34, 30, 32, 35, 37, 40, 42, 45, 48, 51, 54, 57, 60, 63, 66, 70, 74, 77, 81))  # High

_ECC_CODEWORDS_PER_BLOCK = (
    # Phien ban(0 lam phan dem, gia tri nay khong hop le)
    # 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40    Error correction level
    (-1,  7, 10, 15, 20, 26, 18, 20, 24, 30, 18, 20, 24, 26, 30, 22, 24, 28, 30, 28, 28, 28, 28, 30, 30, 26, 28, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30),  # Low
    (-1, 10, 16, 26, 18, 24, 16, 18, 22, 22, 26, 30, 22, 22, 24, 24, 28, 28, 26, 26, 26, 26, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28),  # Medium
    (-1, 13, 22, 18, 26, 18, 24, 18, 22, 20, 24, 28, 26, 24, 20, 30, 24, 28, 28, 26, 30, 28, 30, 30, 30, 30, 28, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30),  # Quartile
    (-1, 17, 28, 22, 16, 22, 28, 26, 26, 24, 28, 24, 28, 22, 24, 24, 30, 28, 28, 26, 28, 30, 24, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30))  # High

_MASK_PATTERNS = (
    (lambda x, y:  (x + y) % 2                  ),
    (lambda x, y:  y % 2                        ),
    (lambda x, y:  x % 3                        ),
    (lambda x, y:  (x + y) % 3                  ),
    (lambda x, y:  (x // 3 + y // 2) % 2        ),
    (lambda x, y:  x * y % 2 + x * y % 3        ),
    (lambda x, y:  (x * y % 2 + x * y % 3) % 2  ),
    (lambda x, y:  ((x + y) % 2 + x * y % 3) % 2),
)

alphanumeric_table = {
    "0" : 0,
    "1" : 1,
    "2" : 2,
    "3" : 3,
    "4" : 4,
    "5" : 5,
    "6" : 6,
    "7" : 7,
    "8" : 8,
    "9" : 9,
    "A" : 10,
    "B" : 11,
    "C" : 12,
    "D" : 13,
    "E" : 14,
    "F" : 15,
    "G" : 16,
    "H" : 17,
    "I" : 18,
    "J" : 19,
    "K" : 20,
    "L" : 21,
    "M" : 22,
    "N" : 23,
    "O" : 24,
    "P" : 25,
    "Q" : 26,
    "R" : 27,
    "S" : 28,
    "T" : 29,
    "U" : 30,
    "V" : 31,
    "W" : 32,
    "X" : 33,
    "Y" : 34,
    "Z" : 35,
    " " : 36,
    "$" : 37,
    "%" : 38,
    "*" : 39,
    "+" : 40,
    "-" : 41,
    "." : 42,
    "/" : 43,
    ":" : 44
}

class QrCode:

    def encode_text(text):
        return QrCode.encode_mid(text)

    def encode_mid(text):
        version = 5
        ecclvl = 2 #mid
        mask = 1
        # 2.1.1 tao chuoi nhi phan
        #  Buoc 1 ma hoa bo chi che do
        modeindicator = "0010" #alphanumeric

        # buoc2 Ma hoa do dai cua du lieu
        # chuyen chieu dai chuoi thanh bin roi shift right khi du 9 bit
        # 9 bit vi mode la alphanumeric va version trong khoang 1 toi 9
        datacodeword = np.binary_repr(len(text), width=9) 
        datacodeword = modeindicator + datacodeword

        # buoc 3 ma hoa du lieu
        #chia chuoi thanh cac doan 2 chu cai
        n = 2
        text2char = [text[i:i+n] for i in range(0, len(text), n)]

        # encode dang alphanumeric
        for i in text2char:
            # tao chuoi nhi phan cho moi cap voi do dai chuoi nhi phan la 11 bit
            if len(i)==2:
                tempInt = alphanumeric_table[i[0]] * 45 + alphanumeric_table[i[1]]
                datacodeword += np.binary_repr(tempInt, width=11)
            else :
                tempInt = alphanumeric_table[i[0]]
                datacodeword += np.binary_repr(tempInt, width=6)

        # buoc 4 hoan thanh cac bit
        #them bit vao datacodeword
        numOfBitReq = 86 * 8
        datacodeword += "0" * min(4,(numOfBitReq - len(datacodeword)))

        # Buoc 5 gioi han chuoi ...
        datacodeword += "0" * (8-len(datacodeword)%8)
        
        # Buoc 6 them cac tu vao cuoi chuoi
        # Pad xen ke cac byte khi dat den dung luong du lieu
        for padbyte in itertools.cycle(("11101100","00010001")):
            if len(datacodeword) >= numOfBitReq:
                break
            datacodeword += padbyte

        return QrCode(version,ecclvl,datacodeword,mask)


    def __init__(self, version, ecclvl, datacodeword, mask):
        self._version = version
        self._size = self._version * 4 + 17
        # 2.1.2 tao bo ma sua loi
        # buoc 1 chon mua sua loi
        self._errcorlvl = ecclvl

        # buoc 2 tao thong diep da thuc
        n = 8
        datacodewords = [datacodeword[i:i+n] for i in range(0, len(datacodeword), n)]
        datacodewords = [int(i,2) for i in datacodewords]

        # khoi tao mang du lieu voi false la mau den, true la mau trang
        self._data = [[False for i in range(self._size)] for i in range(self._size)]  # Khoi dau voi full trang
		# Cho biet cac module chuc nang khong mat na duoc. Huy khi ket thuc ham tao.
        self._isfunction = [[False for i in range(self._size)] for i in range(self._size)]

        # buoc 3 tao bo tao da thuc
        codewords = self._add_ecc_and_interleave(datacodewords)

        # 2.1.3 chon mau mat na
        self._draw_function_pattern()
        self._draw_codewords(codewords)
        self._masking(mask)
        self._draw_format_bits(mask)
        self._mask = mask
        self.makePicfromArray()
        del self._isfunction
    
    def _masking(self, mask):
        masker = _MASK_PATTERNS[mask]
        for y in range(self._size):
            for x in range(self._size):
                self._data[y][x] ^= (masker(x,y)==0) and (not self._isfunction[y][x])

    def _draw_codewords(self, codewords):
        assert len(codewords) == self._get_Num_Raw_Data_Modules(self._version) // 8
        i = 0  
        for right in range(self._size - 1, 0, -2):  
            if right <= 6:
                right -= 1
            for vert in range(self._size):  
                for j in range(2):
                    x = right - j  
                    upward = (right + 1) & 2 == 0
                    y = (self._size - 1 - vert) if upward else vert  
                    if not self._isfunction[y][x] and i < len(codewords) * 8:
                        self._data[y][x] = _get_bit(codewords[i >> 3], 7 - (i & 7))
                        i += 1
        assert i == len(codewords) * 8


    def _draw_function_pattern(self):

        #  ve cac timming patterns van thiet cua qr code
        for i in range(self._size):
            self._set_function_module(6, i, i % 2 == 0)
            self._set_function_module(i, 6, i % 2 == 0)

        #  buoc 2 them thong tin loai
        self._draw_position_detection_maker(3,3)
        self._draw_position_detection_maker(self._size - 4, 3)
        self._draw_position_detection_maker(3, self._size - 4)
        
        numalight = self._version + 2 
        self._draw_alignment_pattern(self._size-numalight,self._size-numalight)
        self._draw_format_bits(0)
        # Buoc 2 them thong tin phien ban
        # do version 5 nen khong them thong tin phien ban
        

    def _draw_format_bits(self, mask):
        data = mask
        rem = data
        for _ in range(10):
            rem = ((rem<<1) ^(rem >>9) *0x537)
        bits = (data <<10 |rem) ^ 0x5412
        assert bits >>15 == 0

        for i in range(0,6):
            self._set_function_module(8, i, _get_bit(bits,i))
        self._set_function_module(8, 7, _get_bit(bits, 6))
        self._set_function_module(8, 8, _get_bit(bits, 7))
        self._set_function_module(7, 8, _get_bit(bits, 8))
        
        for i in range(9,15):
            self._set_function_module(14-i, 8, _get_bit(bits,i))

        for i in range(0,8):
            self._set_function_module(self._size -1 -i, 8 , _get_bit(bits,i))
        for i in range(8,15):
            self._set_function_module(8, self._size -15 + i, _get_bit(bits,i))
        self._set_function_module(8, self._size-8, True)

    def _draw_alignment_pattern(self,x,y):
        """ve alignmentPattern 5*5 voi x,y la vi tri trung tam"""
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                self._set_function_module(x+dx,y+dy,max(abs(dx), abs(dy)) != 1)

    def _draw_position_detection_maker (self, x, y):
        """them position maker vao list voi dau vao la list va size,
        bang cach ve maker 9*9 voi diem giua la x,y"""
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                xx, yy = x + dx, y + dy
                if (0 <= xx < self._size) and (0 <= yy < self._size):
                    # Chebyshev/infinity norm
                    self._set_function_module(xx, yy, max(abs(dx), abs(dy)) not in (2, 4))

    def _get_Num_Raw_Data_Modules(self, ver):
        if not (1 <= ver <= 40):
            raise ValueError("Version number out of range")
        result = (16 * ver + 128) * ver + 64
        if ver >= 2:
            numalign = ver // 7 + 2
            result -= (25 * numalign - 10) * numalign - 55
            if ver >= 7:
                result -= 36
        assert 208 <= result <= 29648
        return result

    def _add_ecc_and_interleave(self, datacodeword):
        numOfBlocks = _NUM_ERROR_CORRECTION_BLOCKS[1][5] # ECC lev M va ver 5
        blockECClen = _ECC_CODEWORDS_PER_BLOCK[1][5]
        rawCodeWords = self._get_Num_Raw_Data_Modules(5)//8
        numshortblocks = numOfBlocks - (rawCodeWords % numOfBlocks)
        shortblocklen = (rawCodeWords//numOfBlocks)

        blocks = []
        rsdiv = self.reedSolomonComputeDivisor(blockECClen)
        k = 0
        for i in range(numOfBlocks):
            dat = datacodeword[k : k + shortblocklen - blockECClen + (0 if i < numshortblocks else 1)]
            k += len(dat)
            ecc = self.reedSolomonComputeRemainder(dat,rsdiv)
            if i < numshortblocks:
                dat.append(0)
            blocks.append(dat+ecc)
        assert k == len(datacodeword)

        result = []
        for i in range(len(blocks[0])):
            for (j,blk) in enumerate(blocks):
                if i!= shortblocklen - blockECClen or j >= numshortblocks:
                    result.append(blk[i])
        assert len(result) == rawCodeWords
        return result

    def reedSolomonMultiply(self,x,y):
        if x >> 8 != 0 or y >> 8 != 0:
            raise ValueError("Byte out of range")
        z = 0
        for i in reversed(range(8)):
            z = (z << 1) ^ ((z >> 7) * 0x11D)
            z ^= ((y >> i) & 1) * x
        assert z >> 8 == 0
        return z

    def reedSolomonComputeDivisor(self,degree):
        if not (1 <= degree <= 255):
            raise ValueError("Degree out of range")

        result = [0] * (degree - 1) + [1]  

        root = 1
        for _ in range(degree):  
            for j in range(degree):
                result[j] = self.reedSolomonMultiply(result[j], root)
                if j + 1 < degree:
                    result[j] ^= result[j + 1]
            root = self.reedSolomonMultiply(root, 0x02)
        return result

    def reedSolomonComputeRemainder(self, data, divisor):
        result = [0] * len(divisor)
        for b in data:  # Polynomial division
            factor = b ^ result.pop(0)
            result.append(0)
            for (i, coef) in enumerate(divisor):
                result[i] ^= self.reedSolomonMultiply(coef, factor)
        return result	

    def _set_function_module(self,x,y, isblack)-> None:
        assert type(isblack) is bool
        self._data[y][x] = isblack
        self._isfunction[y][x] = True

    def makePicfromArray(self):
        """tao anh QRcode tu list"""
        for i in range(self._size):
            for j in range(self._size):
                self._data[i][j] = (self._data[i][j]==False)
        nparr = np.array(self._data)
        image2 = Image.fromarray(nparr)
        image2 = image2.resize((500,500))
        image2.show()
        image2.save('QRcode.png')


def _get_bit(x,i):
    return (x >> i) & 1 != 0

text = "HELLO WORLD 123"
qrcode = QrCode.encode_text(text)
