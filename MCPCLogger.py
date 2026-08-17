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
from collections import Counter, defaultdict

# Folder for debug output files
debugFolder = "/tmp/share/debug"

start = perf_counter()

def format_counter(counter_obj):
    # Sorts by keys for clean, predictable log outputs
    try:
        sorted_items = sorted(counter_obj.items())
    except TypeError:
        # Fallback if keys aren't directly comparable (like tuples of shapes vs strings)
        sorted_items = counter_obj.items()
    return ", ".join([f"{count}x {key}" for key, count in sorted_items])

def process(connection, config, metadata):
    send_original = bool(mrdhelper.get_json_config_param(config, 'sendOriginal', default=False, type='bool'))
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
    
    # grab almost all required parameters once from metadata.
    number_of_coils = metadata.acquisitionSystemInformation.receiverChannels
    TEs = metadata.sequenceParameters.TE
    echo_numbers = len(TEs)
    slices = metadata.encoding[0].encodedSpace.matrixSize.z
    height = metadata.encoding[0].encodedSpace.matrixSize.y
    width = metadata.encoding[0].encodedSpace.matrixSize.x

    # For ISMRMRD Header
    metaVariables = [number_of_coils,TEs,echo_numbers,slices,height,width]
    
    # For items in connection
    coils_set = Counter()
    echo_numbers_set = Counter()
    number_in_series_set = Counter()
    slice_set = Counter()
    data_shape_set = Counter()
    sequenceDesc_set = Counter()
    item_type_set = Counter()
    image_comments_set = Counter()
    image_history_set = Counter()
    FrameOfRef_set = Counter()

    coil_64_meta_counters = defaultdict(Counter)
    coil_64_ice_counters = defaultdict(Counter)


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
                
                if coil !="AC": # Discards all incoming images that are combined.
                    # --- NEW: LAST COIL INSPECTION ---
                    if coil == number_of_coils:
                        meta = ismrmrd.Meta.deserialize(item.attribute_string)
                        for key, value in meta.items():
                            if key == 'IceMiniHead':
                                try:
                                    decoded_ice = base64.b64decode(value).decode('utf-8')
                                    # Parse typical IceMiniHead format (Key = Value \n)
                                    for line in decoded_ice.split('\n'):
                                        if '=' in line:
                                            ice_key, ice_val = line.split('=', 1)
                                            coil_64_ice_counters[ice_key.strip()].update([ice_val.strip()])
                                except Exception as e:
                                    logging.error(f"Failed to decode/parse IceMiniHead for coil 64: {e}")
                            else:
                                # Convert values to string so unhashable types (like lists) don't break the Counter
                                coil_64_meta_counters[key].update([str(value)])
                    # -------------------------------
                    
                    echo_number, number_in_series = sort.get_EchoNo_and_NumberInSeries(item)
                    slice_no = item.slice
                    data_shape = np.stack(item.data).shape
                    sequenceDesc = sort.get_SequenceDescription(item)
                    ImageComment = sort.get_ImageComments(item)

                    meta = ismrmrd.Meta.deserialize(item.attribute_string)
                    ImageHistory = str(meta.get("ImageHistory", "NA"))
                    FrameOfRef = str(meta.get("FrameOfReference", "NA"))
                    
                    

                    item_type = "unknown"
                    if item.image_type is ismrmrd.IMTYPE_PHASE:
                        item_type = "P"
                    elif item.image_type is ismrmrd.IMTYPE_MAGNITUDE:
                        item_type = "M"


                    coils_set.update([coil])
                    echo_numbers_set.update([echo_number])
                    number_in_series_set.update([number_in_series])
                    slice_set.update([slice_no])
                    data_shape_set.update([data_shape])
                    sequenceDesc_set.update([sequenceDesc])
                    image_comments_set.update([ImageComment])
                    item_type_set.update([item_type])

                    image_history_set.update([ImageHistory])
                    FrameOfRef_set.update([FrameOfRef])

                if send_original:
                    tmpMeta = ismrmrd.Meta.deserialize(item.attribute_string)
                    tmpMeta['Keep_image_geometry']    = 1
                    item.attribute_string = tmpMeta.serialize()

                    connection.send_image(item)
                    continue

            elif item is None:
                # print all logs
                logging.debug("----------- ISMRMRD HEADER VARIABLES ----------------------------")
                logging.debug(f"MCPC-3D-S Logger: Number of Coils: {metaVariables[0]} type:{type(metaVariables[0])}")
                logging.debug(f"MCPC-3D-S Logger: TEs: {metaVariables[1]} type:{type(metaVariables[1])}")
                logging.debug(f"MCPC-3D-S Logger: Number of Echo Numbers: {metaVariables[2]} type:{type(metaVariables[2])}")
                logging.debug(f"MCPC-3D-S Logger: Length of Slices: {metaVariables[3]} type:{type(metaVariables[3])}")
                logging.debug(f"MCPC-3D-S Logger: Length of Y: {metaVariables[4]} type:{type(metaVariables[4])}")
                logging.debug(f"MCPC-3D-S Logger: Length of X: {metaVariables[5]} type:{type(metaVariables[5])}")
                logging.debug("----------- ISMRMRD HEADER VARIABLES END ------------------------")
                logging.debug("----------- ITEM VARIABLES ----------------------------")
                logging.debug(f"MCPC-3D-S Logger: Set of Coils: {format_counter(coils_set)}")
                logging.debug(f"MCPC-3D-S Logger: Set of Echo Numbers: {format_counter(echo_numbers_set)}")
                logging.debug(f"MCPC-3D-S Logger: Set of NumberInSeries: {format_counter(number_in_series_set)}")
                logging.debug(f"MCPC-3D-S Logger: Set of Slices: {format_counter(slice_set)}")
                logging.debug(f"MCPC-3D-S Logger: Set of Image Data Shape: {format_counter(data_shape_set)}")
                logging.debug(f"MCPC-3D-S Logger: Set of Image Types: {format_counter(item_type_set)}")
                logging.debug(f"MCPC-3D-S Logger: Set of Sequence Descriptions: {format_counter(sequenceDesc_set)}")
                logging.debug(f"MCPC-3D-S Logger: Set of Image Comments: {format_counter(image_comments_set)}")
                logging.debug(f"MCPC-3D-S Logger: Set of Image History: {format_counter(image_history_set)}")
                logging.debug(f"MCPC-3D-S Logger: Set of FrameOfReference: {format_counter(FrameOfRef_set)}")
                logging.debug("----------- ITEM VARIABLES END ------------------------")

                logging.debug("----------- LAST COIL META DIFFERENCES --------------------")
                for k, v_counter in coil_64_meta_counters.items():
                    diff_flag = "*** DIFFERENCE *** " if len(v_counter) > 1 else ""
                    logging.debug(f"Last Coil Meta [{k}]: {diff_flag}{format_counter(v_counter)}")
                    
                logging.debug("----------- LAST COIL ICE MINI HEAD DIFFERENCES -----------")
                for k, v_counter in coil_64_ice_counters.items():
                    diff_flag = "*** DIFFERENCE *** " if len(v_counter) > 1 else ""
                    logging.debug(f"Last Coil IceMiniHead [{k}]: {diff_flag}{format_counter(v_counter)}")

                break

            else:
                raise Exception("Unsupported data type %s", type(item).__name__)

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