/*****************************************************************************
 * This file is part of the Linux Camera Tool 
 * Copyright (c) 2020 Leopard Imaging Inc.
 * 
 * This program is free software: you can redistribute it and/or modify  
 * it under the terms of the GNU General Public License as published by  
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but 
 * WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License 
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 * 
 * This is the sample code for Leopard USB3.0 camera, hardware.h stores macros
 * for spefic camera tool build. Please refer to the explanation below for 
 * customize build. 
 *
 * Author: Danyu L
 * Last edit: 2019/08
*****************************************************************************/
#pragma once

//#define HAVE_OPENCV_CUDA_SUPPORT

/**
 * rgb gain and offset limits
 */
#define MAX_RGB_GAIN          (2000)
#define MIN_RGB_GAIN          (1)
#define GAIN_FACTOR           (256)
#define MAX_RGB_OFFSET        (255)
#define MIN_RGB_OFFSET        (-255)


#define MAX_RGB_MATRIX        (5000)
#define MIN_RGB_MATRIX        (-5000)
#define DEF_RGB_MATRIX_EDGE   (0)
#define DEF_RGB_MATRIX_DIAG   (256)

#define GAIN_FACTOR           (256)

/**
 * software AE limits
 * you can adjust these values to better fit your sensor
 */
#define MIN_EXP_TIME          (50)
#define MAX_EXP_TIME_FACTOR   (4)
#define MIN_GAIN              (8)
#define MAX_GAIN              (63)
#define TARGET_MEAN_FACTOR    (1.2)
#define ROI_SIZE              (256)

/**
 * OpenCV text scale 
 */
#define TEXT_SCALE_BASE       (50)
/**
 * ISP control range
 */
#define ALPHA_MAX             (5)
#define BETA_MAX              (125)
#define SHARPNESS_MAX         (20)
#define LOW_THRESHOLD_MAX     (100)

/**
 * Default VID for Leopard Imaging USB3 camera 
 * used for udev loop through for recognizing Leopard device
 */ 
#define FX3_USB3_VID              (0x2a0b)
#define OV580_ST_VID              (0x05a9)
#define OV580_OV4689_VID          (0x2b03)     
#define UNKNOWN_LI_VID1           (0x29ab)
#define UNKNOWN_LI_VID2           (0x1e4e)
#define UNKNOWN_LI_VID3           (0x0568)
#define UNKNOWN_LI_VID4           (0x05a3)

#define OV580_OV9271_PID          (0x0581)
#define OV580_OV7251_PID          (0x0581)
#define OV580_OG01A1B_PID         (0x0583)
#define OV580_OV4689_PID          (0x0580)

/**
 * --------------------------camera specific test functions-------------------
 * The following macros are for specific cameras testing function. 
 * Uncomment the specific one if you are using that one for demo
 */
// #define AP0202_WRITE_REG_ON_THE_FLY
// #define AP0202_WRITE_REG_IN_FLASH
// #define OS05A20_PTS_QUERY
// #define AR0231_MIPI_TESTING
// #define IMX334_MONO_MIPI_TESTING


/** 
 * -------------------------- 8-bit I2C slave address list -------------------
 * Put a list here so you don't ask me what is the slave address.
 * Note that for different hardware and revsion, we might have different slave
 * address for same sensors, so try with the address in comment if possible.
 * If you still have problem to access the sensor registers, you can contact
 * our support for a firmware update since most likely that driver wasn't 
 * enabling generic i2c slave register access.
 * Disclaimer: this list doesn't include all our USB3 slave address,
 * just all the cameras I got in touch before.
 */

/**  
 * On-semi Sensor 
 * 16-bit addr, 16-bit value
 * */
#define AP020X_I2C_ADDR         (0xBA)
#define AR0231_I2C_ADDR         (0x20)// 0x30
#define AR0144_I2C_ADDR         (0x20)// 0x30
#define AR0234_I2C_ADDR         (0x20)

/**   
 * Sony Sensor   
 * 16-bit addr, 8-bit value
 */
#define IMX334_I2C_ADDR         (0x34)
#define IMX390_I2C_ADDR         (0x34)
#define IMX324_I2C_ADDR         (0x34)
#define IMX477_I2C_ADDR         (0x20)
/** 
 * Omnivision Sensor
 * 16-bit addr, 8-bit value
 */
#define OV2311_I2C_ADDR         (0xC0)
#define OS05A20_I2C_ADDR        (0x6C)
#define OV5640_I2C_ADDR         (0x78)
#define OV7251_I2C_ADDR         (0xE7)
/** 
 * Toshiba Bridge 
 * 16-bit addr, 16-bit value
 */
#define TC_MIPI_BRIDGE_I2C_ADDR (0x1C)

/**  
 * Maxim Serdes 
 * GMSL1: 8-bit addr, 8-bit value
 * GMSL2: 16-bit addr, 8-bit value
*/
#define MAX96705_SER_I2C_ADDR   (0x80)
#define MAX9295_SER_I2C_ADDR    (0x80) //0x88, 0xC4
#define MAX9272_SER_I2C_ADDR    (0x90)
#define MAX9296_DESER_I2C_ADDR  (0x90) //0xD4

/** 
 * TI Serdes
 * 16-bit addr, 8-bit value 
 * */
#define TI913_SER_I2C_ADDR      (0xB4)//0xB2
#define TI953_SER_I2C_ADDR      (0x60) 
#define TI914_DESER_I2C_ADDR    (0xC0)
#define TI954_DESER_I2C_ADDR    (0x30)

/**
 * --------------------------for nbuf numbers---------------------------------
 * Currently, default nbufs = 2 so the frame rate is higher.
 * Increasing nbufs will results in longer delay for ctrls like 
 * exposure, gain since mmap will allocate buffers to cover more 
 * frames.
 */
#define V4L_BUFFERS_DEFAULT	    (2) 
#define V4L_BUFFERS_MAX	        (32)


/** --- for LI_XU_GENERIC_I2C_RW --- */
#define GENERIC_REG_WRITE_FLG   (0x80)
#define GENERIC_REG_READ_FLG    (0x00)

typedef enum
{
   CROPPED_WIDTH = 1280,
   CROPPED_HEIGHT = 720
} cropped_resolution;
