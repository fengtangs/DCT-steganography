import os
from os import listdir

import cv2
import struct
import bitstring
import numpy  as np
import zigzag as zz
import image_preparation as img
import data_embedding as stego

class DCT():
          
    def encoding(self, COVER_IMAGE_FILEPATH, SECRET_MESSAGE_STRING,flag,count,j):
        self.COVER_IMAGE_FILEPATH = COVER_IMAGE_FILEPATH
        self.SECRET_MESSAGE_STRING = SECRET_MESSAGE_STRING

        NUM_CHANNELS = 3
        # os.chdir("CIFAR")
        # os.chdir(str(j))
        raw_cover_image = cv2.imread(COVER_IMAGE_FILEPATH, flags=cv2.IMREAD_COLOR)
        height, width   = raw_cover_image.shape[:2]
        # Force Image Dimensions to be 8x8 compliant
        while(height % 8): height += 1 # Rows
        while(width  % 8): width  += 1 # Cols
        valid_dim = (width, height)
        padded_image    = cv2.resize(raw_cover_image, valid_dim)
        cover_image_f32 = np.float32(padded_image)
        # cover_image_YCC = img.YCC_Image(cv2.cvtColor(cover_image_f32, cv2.COLOR_BGR2YCrCb))
        cover_image_YCC =   img.YCC_Image(cv2.cvtColor(cover_image_f32, cv2.COLOR_BGR2RGB))
        # cover_image_YCC = img.YCC_Image(cover_image_f32)

        # Placeholder for holding stego image data
        stego_image = np.empty_like(cover_image_f32)

        for chan_index in range(NUM_CHANNELS):
            # FORWARD DCT STAGE
            dct_blocks = [cv2.dct(block) for block in cover_image_YCC.channels[chan_index]]

            # QUANTIZATION STAGE
            dct_quants = [np.around(np.divide(item, img.JPEG_STD_LUM_QUANT_TABLE)) for item in dct_blocks]

            # Sort DCT coefficients by frequency
            sorted_coefficients = [zz.zigzag(block) for block in dct_quants]
            # Embed data in Luminance layer
            # if (chan_index <3):
            if (flag==0 and chan_index==0):
                # DATA INSERTION STAGE
                secret_data = ""
                for char in SECRET_MESSAGE_STRING.encode('ascii'): secret_data += bitstring.pack('uint:8', char)
                embedded_dct_blocks   = stego.embed_encoded_data_into_DCT(secret_data, sorted_coefficients)
                desorted_coefficients = [zz.inverse_zigzag(block, vmax=8,hmax=8) for block in embedded_dct_blocks]
            elif (flag==1 and chan_index==1):
                # DATA INSERTION STAGE
                secret_data = ""
                for char in SECRET_MESSAGE_STRING.encode('ascii'): secret_data += bitstring.pack('uint:8', char)
                print(secret_data)
                embedded_dct_blocks   = stego.embed_encoded_data_into_DCT(secret_data, sorted_coefficients)
                desorted_coefficients = [zz.inverse_zigzag(block, vmax=8,hmax=8) for block in embedded_dct_blocks]
            elif (flag==2 and chan_index==2):
                # DATA INSERTION STAGE
                secret_data = ""
                for char in SECRET_MESSAGE_STRING.encode('ascii'): secret_data += bitstring.pack('uint:8', char)
                embedded_dct_blocks   = stego.embed_encoded_data_into_DCT(secret_data, sorted_coefficients)
                desorted_coefficients = [zz.inverse_zigzag(block, vmax=8,hmax=8) for block in embedded_dct_blocks]
            elif (flag==3 and chan_index<3):
                # DATA INSERTION STAGE
                secret_data = ""
                for char in SECRET_MESSAGE_STRING.encode('ascii'): secret_data += bitstring.pack('uint:8', char)
                embedded_dct_blocks   = stego.embed_encoded_data_into_DCT(secret_data, sorted_coefficients)
                desorted_coefficients = [zz.inverse_zigzag(block, vmax=8,hmax=8) for block in embedded_dct_blocks]
            else:
                # Reorder coefficients to how they originally were
                desorted_coefficients = [zz.inverse_zigzag(block, vmax=8,hmax=8) for block in sorted_coefficients]

            # DEQUANTIZATION STAGE
            dct_dequants = [np.multiply(data, img.JPEG_STD_LUM_QUANT_TABLE) for data in desorted_coefficients]

            # Inverse DCT Stage
            idct_blocks = [cv2.idct(block) for block in dct_dequants]

            # Rebuild full image channel
            stego_image[:,:,chan_index] = np.asarray(img.stitch_8x8_blocks_back_together(cover_image_YCC.width, idct_blocks))
        #-------------------------------------------------------------------------------------------------------------------#

        # Convert back to RGB (BGR) Colorspace
        # stego_image_BGR = cv2.cvtColor(stego_image, cv2.COLOR_YCR_CB2BGR)
        stego_image_BGR = cv2.cvtColor(stego_image, cv2.COLOR_RGB2BGR)


        # Clamp Pixel Values to [0 - 255]
        final_stego_image = np.uint8(np.clip(stego_image_BGR, 0, 255))
        # os.chdir("..")
        # os.chdir("..")
        n = "%04d" % count
        if(flag == 0):
            # os.chdir("YY")
            # os.chdir("0")
            cv2.imwrite("0_"+str(j)+"_"+str(n)+"Y.jpg", final_stego_image)
            # final_stego_image.save(str(count)+".png")
            os.chdir("..")
        if (flag == 1):
            # os.chdir("YY/")
            # os.chdir("0")
            cv2.imwrite("0_"+str(j)+"_"+str(n)+"Cr.jpg", final_stego_image)
            # final_stego_image.save(str(count)+".png")
            os.chdir("..")
        if (flag == 2):
            # os.chdir("YY/")
            # os.chdir("0")
            cv2.imwrite("0_"+str(j)+"_"+str(n)+"Cb.jpg", final_stego_image)
            # final_stego_image.save(str(count)+".png")
            os.chdir("..")
        if (flag == 3):
            # os.chdir("ALL/")
            # os.chdir("AALL")
            cv2.imwrite("0_"+str(j)+"_"+str(n)+".png", final_stego_image)
            # final_stego_image.save(str(count)+".png")
            os.chdir("..")

# dr = CIFAR.Cifar10DataReader(cifar_folder="E:\Tensorlow\cifar-10-batches-py\\")


count=0
DCT().encoding('2.png', "AAAAA", 2, 1,1)
# d, l = dr.next_train_data()
# for j in range (0,10):
#
#     folder_ad="/content/N-NONE/CIFAR\\"+str(j)
#     imglist=listdir(folder_ad)
#     for i in range (0,100):
#             COVER_PATH = imglist[i]
#             print(COVER_PATH)
#             secret_message = "aaaa"
#             # DCT().encoding(COVER_PATH, secret_message+"a", 3, i,j)
#             if (j == 0):
#                 DCT().encoding(COVER_PATH, secret_message , 1, i, j)
#                 DCT().encoding(COVER_PATH, secret_message , 2, i, j)
#             elif (j == 1):
#                 if(i<50):
#                     DCT().encoding(COVER_PATH, secret_message, 0, i, j)
#                 DCT().encoding(COVER_PATH, secret_message , 2, i, j)
#             elif (j == 2):
#                 if(i<50):
#                     DCT().encoding(COVER_PATH, secret_message, 0, i, j)
#                 DCT().encoding(COVER_PATH, secret_message , 1, i, j)
#             else:
#                 if(i<50):
#                     DCT().encoding(COVER_PATH, secret_message, 0, i, j)
#                 DCT().encoding(COVER_PATH, secret_message , 1, i, j)
#                 DCT().encoding(COVER_PATH, secret_message , 2, i, j)
