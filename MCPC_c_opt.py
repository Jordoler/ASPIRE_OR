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
import custom_scripts.mcpc_sort as sort
import subprocess

from juliacall import Main as jl
jl.seval('include("custom_scripts/MCPC-3D-S_c.jl")')

# Folder for debug output files
debugFolder = "/tmp/share/debug"
COIL_LAST_FILTER_KEY = "ImageHistory" # Options: "FrameOfReference" or "ImageHistory"
TESTING = False

start = perf_counter()

def process(connection, config, metadata):
    log_time = bool(mrdhelper.get_json_config_param(config, 'logTime', default=False, type='bool'))
    send_original = bool(mrdhelper.get_json_config_param(config, 'sendOriginal', default=False, type='bool'))
    #replace_grappa = bool(mrdhelper.get_json_config_param(config, 'replaceGRAPPA', default=False, type='bool'))
    replace_grappa = False # No longer need this feature, TODO: remove from script
    phase2pi_scale = bool(mrdhelper.get_json_config_param(config, 'phase2piScale', default=True, type='bool'))

    logging.info("Config: \n%s", config)

    # Metadata should be MRD formatted header, but may be a string
    # if it failed conversion earlier
    try:
        logging.info("Incoming dataset contains %d encodings", len(metadata.encoding))
        logging.info("First encoding is of type '%s', with a matrix size of (%s x %s x %s) and a field of view of (%s x %s x %s)mm^3", 
            metadata.encoding[0].trajectory, 
            metadata.encoding[0].encodedSpace.matrixSize.x, 
            metadata.encoding[0].encodedSpace.matrixSize.y, 
            metadata.encoding[0].encodedSpace.matrixSize.z, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.x, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.y, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.z)

    except:
        logging.info("Improperly formatted metadata: \n%s", metadata)

    
    # optimized method is to pre-allocate a 5d array with [x,y,z,echo,coil]
    # and populate each iteration of the connection loop. 
    # This eliminates all sorting, the need for excessive double for loops as well as minimizes
    # what is stored in memory.
    
    # grab almost all required parameters once from metadata.
    number_of_coils = metadata.acquisitionSystemInformation.receiverChannels
    TEs = metadata.sequenceParameters.TE 
    TEs = [float(t) for t in TEs]
    echo_numbers = len(TEs)
    slices = metadata.encoding[0].encodedSpace.matrixSize.z
    height = metadata.encoding[0].encodedSpace.matrixSize.y
    width = metadata.encoding[0].encodedSpace.matrixSize.x

    # create empty 5d arrays
    mag_data_5d = np.zeros((width, height, slices, echo_numbers, number_of_coils), dtype=np.float32,order="F")
    ph_data_5d  = np.zeros((width, height, slices, echo_numbers, number_of_coils), dtype=np.float32,order="F")

    # set management for 5d arrays in cases where if int(coil) == number_of_coils block fails
    filled_ph_slices = set()
    filled_mag_slices = set()

    # head and meta data array template used in populating image head/meta data after MCPC-3D-S.
    mag_meta_templates = np.empty((echo_numbers, slices), dtype=object)
    ph_meta_templates = np.empty((echo_numbers, slices), dtype=object)
     
    coil_last_filter_value = None
    
    try:
        for item in connection:
            # ----------------------------------------------------------
            # Raw k-space data messages
            # ----------------------------------------------------------
            if isinstance(item, ismrmrd.Acquisition):
                raise Exception("Raw k-space data is not supported by this module")
            
            # ----------------------------------------------------------
            # Image data messages
            # ----------------------------------------------------------
            elif isinstance(item, ismrmrd.Image):
                coil = sort.get_coil(item)
                meta = ismrmrd.Meta.deserialize(item.attribute_string)
                
                # 1. EXPLICITLY IDENTIFY GRAPPA / COMBINED IMAGES
                # We fetch the filter string to check for any combination images (e.g., Sum-of-Squares, ComplexAdd)
                current_coil_filter = str(meta.get(COIL_LAST_FILTER_KEY, "UNKNOWN"))
                is_grappa_or_combined = coil == "AC" or any(x in current_coil_filter for x in ["SoS", "ComplexAdd", "CC:"])

                # If send_original is True, send everything.
                # If replace_grappa is True, only send it if it's NOT a GRAPPA/combined image.
                if send_original or (replace_grappa and not is_grappa_or_combined):
                    unchanged_slice = sort.format_unchanged(item)
                    connection.send_image(unchanged_slice)

                # 3. FILTER OUT GRAPPA FROM THE MCPC 5D ARRAY
                # We completely discard GRAPPA/combined images from entering the 5D accumulation arrays
                if is_grappa_or_combined:
                    continue 

                # ----------------------------------------------------------
                # 4. POPULATE 5D ARRAYS (Only uncombined data should reach here)
                # ----------------------------------------------------------
                try:
                    echo_number, slice_no = sort.get_EchoNo_and_SliceNo(item)
                except Exception as e:
                    logging.debug(f"Sort_debug: Echo Number/Slice No Fetch Error! on coil: {coil}")
                    logging.error(traceback.format_exc())
                    continue

                echo_id = int(echo_number) - 1
                coil_id = int(coil) - 1
                slice_id = int(slice_no) - 1
                raw_data_2d = item.data[0, 0, :, :].T 

                if item.image_type is ismrmrd.IMTYPE_PHASE:
                    coord = (slice_id, echo_id, coil_id)
                    if coord not in filled_ph_slices:
                        ph_data_5d[:, :, slice_id, echo_id, coil_id] = raw_data_2d
                        filled_ph_slices.add(coord)
                    if coil_id == 0:
                        ph_meta_templates[echo_id, slice_id] = {'head': item.getHead(),
                                                                'meta': meta}#ismrmrd.Meta.deserialize(item.attribute_string)}
            
                elif item.image_type is ismrmrd.IMTYPE_MAGNITUDE:
                    coord = (slice_id, echo_id, coil_id)
                    if coord not in filled_mag_slices:
                        mag_data_5d[:, :, slice_id, echo_id, coil_id] = raw_data_2d
                        filled_mag_slices.add(coord)
                    if coil_id == 0:
                        mag_meta_templates[echo_id, slice_id] = {'head': item.getHead(),
                                                            'meta': meta}#ismrmrd.Meta.deserialize(item.attribute_string)}
                
                
                # if send_original:
                #     unchanged_slice = sort.format_unchanged(item)
                #     connection.send_image(unchanged_slice)
                #     # removed continue to ensure MCPC occurs

            elif item is None:
                here = None
                sort.log_time(log_time,start,"Last item recieved, process took")
                currentSeries = 60 # 60 avoids hitting other series if sendOriginal is true
                TEs.sort()
                logging.debug(f"Sort_debug: TEs - {TEs}")
                logging.debug(f"Sort_debug: Last Coil filter locked on {COIL_LAST_FILTER_KEY} == {coil_last_filter_value}")
                
                logging.debug("Sort_debug: Converting phase from int to radians...")
                here = perf_counter() if log_time else None
                ph_data_5d = sort.ph5d_to_radians_inplace(ph_data_5d, phase2pi_scale)
                sort.log_time(log_time, here, "phase int to radians took")

                logging.debug("Sort_debug: Emulating nifti readphase normalization...")
                here = perf_counter() if log_time else None
                ph_min = ph_data_5d.min()
                ph_max = ph_data_5d.max()
                #ph_data_5d = (ph_data_5d - ph_min) / (ph_max - ph_min) * (2 * np.pi) - np.pi
                range_val = ph_max - ph_min
                ph_data_5d -= ph_min
                ph_data_5d /= range_val
                ph_data_5d *= (2 * np.pi)
                ph_data_5d -= np.pi
                sort.log_time(log_time, here, "nifti readphase normalization took")

                # MCPC warm call stuff
                logging.debug("Sort_debug: Passing 5d arrays to Julia in-memory...")
                here = perf_counter() if log_time else None
                

                combined_mag_4d, combined_ph_4d = jl.combine_coils_in_memory(mag_data_5d,ph_data_5d,TEs)
                sort.log_time(log_time, here, "Julia in-memory computation took")

                combined_mag_4d = np.ascontiguousarray(np.asarray(combined_mag_4d).transpose((3, 2, 1, 0)))
                combined_ph_4d  = np.ascontiguousarray(np.asarray(combined_ph_4d).transpose((3, 2, 1, 0)))


                logging.debug("Sort_debug: processing combined mag images...")
                here = perf_counter() if log_time else None
                combined_mag = process_combined_images_opt(combined_mag_4d,"mag",currentSeries,mag_meta_templates,config)
                sort.log_time(log_time,here,"processing combined mag images took")
                
                logging.debug("Sort_debug: sending combined mag images...")
                here = perf_counter() if log_time else None
                for image in combined_mag:
                    connection.send_image(image)
                sort.log_time(log_time,here,"sending combined mag images took")
                
                logging.debug("Sort_debug: processing combined phase images...")
                here = perf_counter() if log_time else None
                combined_ph = process_combined_images_opt(combined_ph_4d,"ph",currentSeries,ph_meta_templates,config)
                sort.log_time(log_time,here,"processing combined phase images took")
                
                logging.debug("Sort_debug: sending combined phase images...")
                here = perf_counter() if log_time else None
                for image in combined_ph:
                    connection.send_image(image)
                sort.log_time(log_time,here,"sending combined phase images took")
                
                logging.debug("Sort_debug: Config: MCPC_compiled")
                break

            else:
                raise Exception("Unsupported data type %s", type(item).__name__)

    except MemoryError as me:
        error_msg = f"CRITICAL: Process failed due to Memory Overflow (MemoryError)!\n{traceback.format_exc()}"
        logging.critical(error_msg)
        print(f"\n[PROCESS FAILED]: {error_msg}\n")
        
        # Notify the client connection of the exact failure reason if possible
        try:
            connection.send_logging(constants.MRD_LOGGING_ERROR, error_msg)
        except:
            pass
        connection.shutdown_close()

    except Exception as e:
        logging.error(traceback.format_exc())
        connection.send_logging(constants.MRD_LOGGING_ERROR, traceback.format_exc())
        
        # Close connection without sending MRD_MESSAGE_CLOSE message to signal failure
        connection.shutdown_close()

    finally:
        end = perf_counter()
        print(f"Sort_debug: Total elapsed time: {end - start:.6f} seconds")
        try:
            connection.send_close()
        except:
            logging.error("Failed to send close message!")

def process_combined_images_opt(combined_4d,echo_type: str,currentSeries, meta_templates,config):
    phase2pi_scale = bool(mrdhelper.get_json_config_param(config, 'phase2piScale', default=True, type='bool'))

    combined_4d = np.nan_to_num(combined_4d, nan=0.0,copy=False)
    combined_4d = combined_4d.astype(np.int16) if echo_type == "mag" else combined_4d.astype(np.float32)
    
    if echo_type=="mag":
        combined_4d = np.clip(combined_4d, a_min=0, a_max=4095)
    elif echo_type=="ph":
        combined_4d = sort.radians_to_ph_int(combined_4d,phase2pi_scale)
    
    combinedOut = [None] * combined_4d.shape[0]
    for echo_idx in range(combined_4d.shape[0]):
        echo_val = echo_idx + 1 # Clear representation of the current 1-based echo number
        
        imagesOut = [None] * combined_4d.shape[1]
        for slice_idx in range(combined_4d.shape[1]):
            imagesOut[slice_idx] = ismrmrd.Image.from_array(combined_4d[echo_idx, slice_idx, :, :], transpose=False) # (y,x) for the slice_idx slice with echo_idx
            
            template = meta_templates[echo_idx, slice_idx]
            if template is None:
                logging.error(f"E11 - Missing metadata template for echo index {echo_idx}, slice index {slice_idx}")
                continue
            
            oldHeader = template["head"]
            tmpMeta = template["meta"]

            oldHeader.data_type = imagesOut[slice_idx].data_type
            oldHeader.channels = 1 # update from julia multi-echo combination
            
            if (imagesOut[slice_idx].data_type == ismrmrd.DATATYPE_CXFLOAT) or (imagesOut[slice_idx].data_type == ismrmrd.DATATYPE_CXDOUBLE):
                oldHeader.image_type = ismrmrd.IMTYPE_COMPLEX

            # if mrdhelper.get_meta_value(tmpMeta, 'IceMiniHead') is not None:
            #     if mrdhelper.extract_minihead_bool_param(base64.b64decode(tmpMeta[slice_idx]['IceMiniHead']).decode('utf-8'), 'BIsSeriesEnd') is True:
            #         currentSeries += 1
            
            imagesOut[slice_idx].setHead(oldHeader)
            
            maxVal = combined_4d[echo_idx,slice_idx,:,:].max()

            # Create a copy of the original ISMRMRD Meta attributes and update
            tmpMeta['DataRole']                       = 'Image'
            tmpMeta['ImageProcessingHistory']         = ['PYTHON', 'JULIA', 'MCPC-3D-S']
            tmpMeta['WindowCenter']                   = str((maxVal+1)/2)
            tmpMeta['WindowWidth']                    = str((maxVal+1))
            tmpMeta['SequenceDescriptionAdditional']  = f'OPENRECON_MCPC_{echo_type}'
            tmpMeta['Keep_image_geometry']            = 1

            # Add image orientation directions to MetaAttributes if not already present
            if tmpMeta.get('ImageRowDir') is None:
                tmpMeta['ImageRowDir'] = ["{:.18f}".format(oldHeader.read_dir[0]), "{:.18f}".format(oldHeader.read_dir[1]), "{:.18f}".format(oldHeader.read_dir[2])]

            if tmpMeta.get('ImageColumnDir') is None:
                tmpMeta['ImageColumnDir'] = ["{:.18f}".format(oldHeader.phase_dir[0]), "{:.18f}".format(oldHeader.phase_dir[1]), "{:.18f}".format(oldHeader.phase_dir[2])]

            metaXml = tmpMeta.serialize()

            imagesOut[slice_idx].attribute_string = metaXml
        combinedOut[echo_idx] = imagesOut
    return combinedOut
