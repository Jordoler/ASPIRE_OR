"""
A utility module to help sort multi-echo uncombined MRD data streams in an Open Recon container
"""
import ismrmrd
import os
import itertools
import logging
import traceback
import numpy as np
import numpy.fft as fft
import xml.dom.minidom
import base64
import ctypes
import re
import mrdhelper
import nibabel as nib
import constants
from time import perf_counter


def get_coil(item,isTesting=False):
    meta = ismrmrd.Meta.deserialize(item.attribute_string)
    if isTesting:
        return int(meta["channel_id"])
    else:
        try:
            IceMiniHeader = base64.b64decode(meta['IceMiniHead']).decode('utf-8')
            CoilString = str(mrdhelper.extract_minihead_string_param(IceMiniHeader, 'CoilString'))
            if CoilString=="AC":
                return CoilString
            else:
                CoilInt = int(CoilString[1:])
                return CoilInt
        except Exception as e:
            logging.error("E10 - Failed to find CoilString")
            return None

def get_TEs(metadata,TEs,item=None,isTesting=False):
    """
    inputs the metadata from process, and a TEs list,
    returns an appended list of new unique TEs in ascending order

    Usage: TEs = get_TEs(metadata, TEs, item, isTesting) for testing
           TEs = get_TEs(metadata, TEs) for not testing
    """
    try:
        if isTesting:
            meta = ismrmrd.Meta.deserialize(item.attribute_string)
            TEs = TEs + [meta["EchoTime"]]
        else:
            TEs = TEs + list(metadata.sequenceParameters.TE)
    except Exception as e:
        logging.debug(f"E09 - Failed to acquire TEs from metadata \n\n{e}")
    return list(set(TEs))

def get_NumberInSeries(item):
    if not isinstance(item, ismrmrd.Image):
        logging.error("E03 - Error item provided to get_NumberInSeries is not an ismrmrd.Image")
    """
    returns NumberInSeries from image in connection
    """
    meta = ismrmrd.Meta.deserialize(item.attribute_string)
    
    try:
        IceMiniHeader = base64.b64decode(meta['IceMiniHead']).decode('utf-8')
        NumberInSeries = int(mrdhelper.extract_minihead_long_param(IceMiniHeader, 'NumberInSeries'))
        return NumberInSeries
    except Exception as e:
        #logging.error(f"E01 - Failed to acquire NumberInSeries from item:{item.slice}, resorting to item.slice \n{e}")
        return item.slice

def get_EchoNumber(item,isTesting=False):
    if not isinstance(item, ismrmrd.Image):
        logging.error("E04 - Error item provided to get_EchoNumber is not an ismrmrd.Image")
    """
    returns EchoNumber from item in connection
    """
    meta = ismrmrd.Meta.deserialize(item.attribute_string)
    if isTesting:
        return meta["EchoNumber"]
    else:
        try:
            IceMiniHeader = base64.b64decode(meta['IceMiniHead']).decode('utf-8')
            EchoNumber = int(mrdhelper.extract_minihead_long_param(IceMiniHeader, 'EchoNumber'))
            return EchoNumber
        except Exception as e:
            logging.error(f"E02 - Failed to acquire EchoNumber from item:{item.slice}\n{e}")
            return 0
    
def get_EchoNo_and_NumberInSeries(item,isTesting=False):
    """
    used in optimized program
    """
    if not isinstance(item, ismrmrd.Image):
        logging.error("E03 - Error item provided is not an ismrmrd.Image")
    meta = ismrmrd.Meta.deserialize(item.attribute_string)
    if isTesting:
        return meta["EchoNumber"], item.slice
    else:
        try:
            IceMiniHeader = base64.b64decode(meta['IceMiniHead']).decode('utf-8')
            NumberInSeries = int(mrdhelper.extract_minihead_long_param(IceMiniHeader, 'NumberInSeries'))
            EchoNumber = int(mrdhelper.extract_minihead_long_param(IceMiniHeader, 'EchoNumber'))
            return EchoNumber, NumberInSeries
        except Exception as e:
            logging.error(f"E01 - Failed to acquire NumberInSeries from item:{item.slice}, resorting to item.slice \n{e}")
            return None
        
def get_EchoNo_and_SliceNo(item):
    """
    used in optimized program
    """
    if not isinstance(item, ismrmrd.Image):
        logging.error("E03 - Error item provided is not an ismrmrd.Image")
    meta = ismrmrd.Meta.deserialize(item.attribute_string)
    try:
        IceMiniHeader = base64.b64decode(meta['IceMiniHead']).decode('utf-8')
        EchoNumber = int(mrdhelper.extract_minihead_long_param(IceMiniHeader, 'EchoNumber'))
        Slice_No = item.slice
        return EchoNumber, Slice_No
    except Exception as e:
        logging.error(f"E01 - Failed to acquire NumberInSeries from item:{item.slice}, resorting to item.slice \n{e}")
        return None

def sort_echos(dict_images:dict):
    "Sorts the dictionary by ascending echo number"
    dict_images = dict(sorted(dict_images.items()))
    return dict_images

def get_SliceNo(item):
    return item.slice

def sort_images(dict_images:dict,echo_numbers=None):
    """
    Sorts echo dictionaries by NumberInSeries and in ascending order
    does return a variable sorts inputted dictionary
    """
    if not dict_images:
        logging.error("E05 - dict provided to sort_images is Empty")
        return None
    echo_numbers = sorted(list(dict_images.keys())) if echo_numbers is None else echo_numbers
    for echo_number in echo_numbers:
        for coil_id in dict_images[echo_number]:
            dict_images[echo_number][coil_id].sort(key=get_SliceNo)
    return None

def get_5d_array(dict_images:dict,echo_numbers=None):
    """
    !!!WIP!!!
    inputs an echo dictionary and returns a 5d array [x,y,z,echo,coil/channel]
    """
    if not dict_images:
        logging.error("E06 - Empty dictionary provided to process_images")
        return []
    
    echo_numbers = sorted(list(dict_images.keys())) if echo_numbers is None else sorted(echo_numbers)
    
    echo_list = [] # a temporary list that holds [Echo1[imageNo,coil,z,y,x],Echo2-[imageNo,channel,z,y,x]]
    
    #NOTE: below is a double for loop, find a better approach.
    for echo_number in echo_numbers:
        coil_ids = sorted(list(dict_images[echo_number].keys()))
        coil_list = []
        for coil_id in coil_ids:
            # Stack all slices for this specific coil channel
            slice_data_arrays = [image.data for image in dict_images[echo_number][coil_id]]
            coil_data = np.stack(slice_data_arrays, axis=0) 
            coil_data = coil_data[:, 0, 0, :, :] 
            coil_list.append(coil_data)
            
        coils_stacked = np.stack(coil_list, axis=0) # [coil, z, y, x]
        echo_list.append(coils_stacked)
        
    data5d = np.stack(echo_list, axis=0) # [echo, coil, z, y, x]
    logging.debug(f"Sort_debug: data5d.shape: {data5d.shape} - [echo, coil, z, y, x]")
    logging.debug(f"Sort_debug: transposing...")
    # Transpose from [echo, coil, z, y, x] to [x, y, z, echo, coil] for MCPC-3D-S
    data5d = data5d.transpose((4, 3, 2, 0, 1))
    logging.debug(f"Sort_debug: data5d.shape: {data5d.shape} - [x, y, z, echo, coil]")
    return data5d

def get_SequenceDescription(item):
    meta = ismrmrd.Meta.deserialize(item.attribute_string)
    try:
        ImageComments = str(meta["ImageComments"])
        return ImageComments
    except Exception as e:
        logging.error(f"E09 - Failed to acquire ImageComments from item:{item.slice}\n{e}")
        return None

def get_ImageComments(item):
    meta = ismrmrd.Meta.deserialize(item.attribute_string)
    try:
        IceMiniHeader = base64.b64decode(meta['IceMiniHead']).decode('utf-8')
        SeqDesc = str(mrdhelper.extract_minihead_string_param(IceMiniHeader, 'SequenceDescription'))
        return SeqDesc
    except Exception as e:
        logging.error(f"E09 - Failed to acquire SequenceDescription from item:{item.slice}\n{e}")
        return None

def ph5d_to_radians(data5d, is2piScaling=True):
    data_float = data5d.astype(np.float32)
    data_norm = data_float / 4095.0
    
    if is2piScaling:
        data_rad = data_norm * 2 * np.pi
    else:
        data_rad = data_norm * np.pi
    return data_rad

def ph5d_to_radians_inplace(data5d, is2piScaling=True):
    data5d /= 4095.0
    if is2piScaling:
        data5d *= (2 * np.pi)
    else:
        data5d *= np.pi
    return data5d # Returns the same array object

def radians_to_ph_int(data_rad, is2piScaling=False):
    if is2piScaling:
        data_scaled = (data_rad / (2 * np.pi)) * 4095.0
    else:
        data_scaled = (data_rad / np.pi) * 4095.0
    data_int = np.clip(data_scaled, a_min=-4095, a_max=4095)
    return data_int.astype(np.int16)

def data5d_to_Nifti(data5d,echo_type=None):
    echo_type = "undefined" if echo_type is None else echo_type
    affine = np.eye(4) # Might need to find a get_affine method
    nib.save(nib.Nifti2Image(data5d.astype(np.float32),affine),echo_type+"_5d.nii")
    
def format_TEs(TEs: list) -> str:
    return [str(te) for te in TEs]

def format_unchanged(item):
    tmpMeta = ismrmrd.Meta.deserialize(item.attribute_string)
    tmpMeta['Keep_image_geometry']    = 1
    item.attribute_string = tmpMeta.serialize()
    return item

def log_time(flag,start:None, string:str):
    if flag and start is not None:
            now = perf_counter()
            time = f" - {now - start:.6f} seconds"
            logging.debug("Sort_debug: "+string+time)
    return None