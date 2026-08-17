#!/usr/bin/env python3
"""
NIfTI to ISMRMRD Converter
Converts NIfTI files to ISMRMRD format for testing the OpenRecon pipeline
adapted from: https://github.com/neurodesk/neurocontainers/blob/main/recipes/musclemap/nifti2mrd.py
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
import json
from pathlib import Path

try:
    import ismrmrd
    print("✅ Successfully imported ismrmrd module")
except ImportError as e:
    print(f"❌ Failed to import ismrmrd: {e}")
    print("   Creating mock ISMRMRD classes for testing...")
    
    # Create mock ISMRMRD classes for testing
    class MockImage:
        def __init__(self, data):
            self.data = data
            self.meta = {}
            self.attribute_string = ""
            self.image_type = 1
            self.image_index = 0
            self.image_series_index = 1
        
        @classmethod
        def from_array(cls, data, transpose=True):
            return cls(data)
    
    class MockMeta:
        def __init__(self):
            self._data = {}
        
        def __setitem__(self, key, value):
            self._data[key] = value
        
        def __getitem__(self, key):
            return self._data[key]
        
        def get(self, key, default=None):
            return self._data.get(key, default)
        
        def serialize(self):
            return json.dumps(self._data)
    
    import types
    ismrmrd = types.ModuleType('ismrmrd')
    ismrmrd.Image = MockImage
    ismrmrd.Meta = MockMeta
    IMTYPE_MAGNITUDE = 1
    IMTYPE_PHASE = 2


def extract_orientation_from_affine(affine, shape):
    """
    Extract position and direction vectors from NIfTI affine matrix
    """
    print("🧭 Extracting orientation from affine matrix...")
    print(f"   Affine matrix:\n{affine}")
    
    rotation_scale = affine[:3, :3]
    translation = affine[:3, 3]
    
    col0 = rotation_scale[:, 0]
    col1 = rotation_scale[:, 1]
    col2 = rotation_scale[:, 2]
    
    voxel_size_x = np.linalg.norm(col0)
    voxel_size_y = np.linalg.norm(col1)
    voxel_size_z = np.linalg.norm(col2)
    
    print(f"   Voxel sizes from affine: [{voxel_size_x:.4f}, {voxel_size_y:.4f}, {voxel_size_z:.4f}] mm")
    
    read_dir = col0 / voxel_size_x if voxel_size_x > 0 else col0
    phase_dir = col1 / voxel_size_y if voxel_size_y > 0 else col1
    slice_dir = col2 / voxel_size_z if voxel_size_z > 0 else col2
    
    position = translation.copy()

    # NIfTI affine is in RAS; MRD/DICOM uses LPS. Convert by negating x and y.
    ras_to_lps = np.array([-1, -1, 1], dtype=float)
    read_dir  = read_dir  * ras_to_lps
    phase_dir = phase_dir * ras_to_lps
    slice_dir = slice_dir * ras_to_lps
    position  = position  * ras_to_lps

    print("🔄 Converted from RAS to LPS...")
    print(f"   Position: [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}] mm")
    print(f"   Read direction:  [{read_dir[0]:.4f}, {read_dir[1]:.4f}, {read_dir[2]:.4f}]")
    print(f"   Phase direction: [{phase_dir[0]:.4f}, {phase_dir[1]:.4f}, {phase_dir[2]:.4f}]")
    print(f"   Slice direction: [{slice_dir[0]:.4f}, {slice_dir[1]:.4f}, {slice_dir[2]:.4f}]")
    
    return {
        'position': position.tolist(),
        'read_dir': read_dir.tolist(),
        'phase_dir': phase_dir.tolist(),
        'slice_dir': slice_dir.tolist(),
        'voxel_size': [voxel_size_x, voxel_size_y, voxel_size_z]
    }


def extract_metadata_from_json(nifti_path):
    """Extract metadata from the accompanying JSON file"""
    # Build JSON path intelligently
    json_path = nifti_path.replace('.nii.gz', '.json').replace('.nii', '.json')
    print(f"🏷️ Extracting metadata from JSON: {json_path}")
    
    # Core defaults
    metadata = {
        'config': 'openrecon',
        'enable_measurements': True,
        'enable_reporting': True,
        'confidence_threshold': 0.5,
        'PatientName': 'TEST^PATIENT',
        'StudyDescription': 'OPENRECON TEST',
        'SeriesDescription': 'TEST_SERIES',
        'PatientID': 'TESTPAT001',
        'SeriesNumber': 1
    }
    
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                json_data = json.load(f)
                
            # Update all metadata fields directly from JSON
            metadata.update(json_data)
            print(f"✅ Parsed JSON metadata successfully. Loaded {len(json_data)} keys.")
        except Exception as e:
            print(f"⚠️ Warning: Could not parse JSON file: {e}")
            print("   Using default metadata values")
    else:
        print(f"⚠️ Warning: JSON file not found at {json_path}")
        print("   Using default metadata values")
        
    return metadata


def convert_nifti_to_ismrmrd(nifti_path, output_path=None):
    """Convert NIfTI file to ISMRMRD format"""
    
    if not os.path.exists(nifti_path):
        raise FileNotFoundError(f"NIfTI file not found: {nifti_path}")
    
    print(f"🔄 Converting NIfTI to ISMRMRD format")
    print(f"   Input: {nifti_path}")
    
    # Load NIfTI data
    print("📖 Loading NIfTI file...")
    nii = nib.load(nifti_path)
    data = nii.get_fdata()
    affine = nii.affine
    
    print(f"📐 Original data shape: {data.shape}")
    print(f"🔢 Value range: {data.min():.2f} - {data.max():.2f}")
    print(f"📊 Data type: {data.dtype}")
    
    # Extract orientation information from affine matrix
    orientation_info = extract_orientation_from_affine(affine, data.shape)
    
    if data.max() > 4095:
        data = (data / data.max()) * 4095
        print(f"🔧 Normalized data to range: {data.min():.2f} - {data.max():.2f}")
    
    # Ensure standard 4D [X, Y, Z, Channels] block
    if len(data.shape) == 2:
        data = data[:, :, np.newaxis, np.newaxis]
        print(f"📝 Expanded 2D to 4D: {data.shape}")
    elif len(data.shape) == 3:
        data = data[:, :, :, np.newaxis]
        print(f"📝 Expanded 3D to 4D: {data.shape}")
        
    num_x, num_y, num_z, num_c = data.shape
    
    print("🏗️ Creating ISMRMRD Image object...")
    if np.iscomplexobj(data):
        ismrmrd_data = np.abs(data).astype(np.float32)
        print("🔧 Converted complex data to magnitude (float32)")
    else:
        ismrmrd_data = data.astype(np.float32)
    
    # canonical layout mapped to [Channels, Z, Y, X] for the global image block
    ismrmrd_volume = ismrmrd_data.transpose((3, 2, 1, 0))
    try:
        ismrmrd_image = ismrmrd.Image.from_array(ismrmrd_volume, transpose=False)
    except TypeError:
        ismrmrd_image = ismrmrd.Image.from_array(ismrmrd_volume)
    
    # Extract metadata using the JSON parsing function
    metadata = extract_metadata_from_json(nifti_path)

    # Determine image type based on JSON
    determined_image_type = IMTYPE_MAGNITUDE if 'IMTYPE_MAGNITUDE' in globals() else 1
    if 'ImageType' in metadata:
        if "P" in metadata['ImageType'] or "PHASE" in metadata['ImageType']:
            determined_image_type = IMTYPE_PHASE if 'IMTYPE_PHASE' in globals() else 2
            print("🔍 Detected PHASE image from metadata")
        elif "M" in metadata['ImageType']:
            print("🔍 Detected MAGNITUDE image from metadata")
            
    if hasattr(ismrmrd_image, 'image_type'):
        ismrmrd_image.image_type = determined_image_type
    if hasattr(ismrmrd_image, 'image_series_index'):
        ismrmrd_image.image_series_index = 1
    if hasattr(ismrmrd_image, 'image_index'):
        ismrmrd_image.image_index = 0
    
    metadata['position'] = orientation_info['position']
    metadata['read_dir'] = orientation_info['read_dir']
    metadata['phase_dir'] = orientation_info['phase_dir']
    metadata['slice_dir'] = orientation_info['slice_dir']
    
    voxel_size = orientation_info['voxel_size']
    
    field_of_view = [
        num_x * voxel_size[0],  
        num_y * voxel_size[1],  
        num_z * voxel_size[2]   
    ]
    
    metadata['PixelSpacing'] = [voxel_size[1], voxel_size[0]]
    metadata['SliceThickness'] = voxel_size[2]
    metadata['field_of_view'] = field_of_view
    
    if hasattr(ismrmrd_image, 'meta'):
        ismrmrd_image.meta = metadata
    
    if hasattr(ismrmrd_image, 'field_of_view'):
        ismrmrd_image.field_of_view[:] = field_of_view
    
    if hasattr(ismrmrd_image, 'position'):
        ismrmrd_image.position[:] = orientation_info['position']
    
    if hasattr(ismrmrd_image, 'read_dir'):
        ismrmrd_image.read_dir[:] = orientation_info['read_dir']
    
    if hasattr(ismrmrd_image, 'phase_dir'):
        ismrmrd_image.phase_dir[:] = orientation_info['phase_dir']
    
    if hasattr(ismrmrd_image, 'slice_dir'):
        ismrmrd_image.slice_dir[:] = orientation_info['slice_dir']
    
    meta_obj = ismrmrd.Meta()
    for key, value in metadata.items():
        if isinstance(value, (list, tuple)):
            meta_obj[key] = list(value)
        else:
            meta_obj[key] = str(value)
    
    meta_obj['DataRole'] = 'Image'
    meta_obj['ImageProcessingHistory'] = ['NIfTI_CONVERSION']
    meta_obj['Keep_image_geometry'] = 1
    meta_obj['orientation_extracted'] = 'true'
    
    if hasattr(ismrmrd_image, 'attribute_string'):
        ismrmrd_image.attribute_string = meta_obj.serialize()
    
    print(f"✅ Successfully created ISMRMRD Image Structure")
    
    if output_path:
        print(f"💾 Saving to: {output_path}")
        
        # Check if the file exists to determine if we are appending
        file_exists = os.path.exists(output_path)
        if file_exists:
            print(f"📎 Appending to existing file: {output_path}")
        else:
            print(f"📝 Creating new file: {output_path}")
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        try:
            if hasattr(ismrmrd, 'Dataset'):
                # The create_if_needed flag natively opens existing files in append mode
                mrdDset = ismrmrd.Dataset(output_path, 'dataset', create_if_needed=True)
                mrdDset._file.require_group('dataset')
                
                # Only construct and write the global XML header if the file is new
                # If we are appending, we preserve the existing study-level XML header
                if not file_exists:
                    mrdHead = ismrmrd.xsd.ismrmrdHeader()
                    
                    mrdHead.studyInformation = ismrmrd.xsd.studyInformationType()
                    mrdHead.studyInformation.studyDescription = metadata.get('StudyDescription', 'NIFTI_CONVERSION')
                    
                    mrdHead.subjectInformation = ismrmrd.xsd.subjectInformationType()
                    mrdHead.subjectInformation.patientName = metadata.get('PatientName', 'TEST^PATIENT')
                    mrdHead.subjectInformation.patientID = metadata.get('PatientID', 'TEST001')
                    
                    mrdHead.acquisitionSystemInformation = ismrmrd.xsd.acquisitionSystemInformationType()
                    mrdHead.acquisitionSystemInformation.systemVendor = 'NIfTI_Converter'
                    mrdHead.acquisitionSystemInformation.systemModel = 'Virtual'
                    mrdHead.acquisitionSystemInformation.institutionName = 'Test'
                    
                    encoding = ismrmrd.xsd.encodingType()
                    encoding.trajectory = ismrmrd.xsd.trajectoryType.CARTESIAN
                    
                    encoding.encodedSpace = ismrmrd.xsd.encodingSpaceType()
                    encoding.encodedSpace.matrixSize = ismrmrd.xsd.matrixSizeType()
                    encoding.encodedSpace.matrixSize.x = int(num_x)
                    encoding.encodedSpace.matrixSize.y = int(num_y)
                    encoding.encodedSpace.matrixSize.z = 1
                    
                    encoding.encodedSpace.fieldOfView_mm = ismrmrd.xsd.fieldOfViewMm()
                    encoding.encodedSpace.fieldOfView_mm.x = float(num_x * voxel_size[0])
                    encoding.encodedSpace.fieldOfView_mm.y = float(num_y * voxel_size[1])
                    encoding.encodedSpace.fieldOfView_mm.z = float(voxel_size[2])
                    
                    encoding.reconSpace = ismrmrd.xsd.encodingSpaceType()
                    encoding.reconSpace.matrixSize = ismrmrd.xsd.matrixSizeType()
                    encoding.reconSpace.matrixSize.x = int(num_x)
                    encoding.reconSpace.matrixSize.y = int(num_y)
                    encoding.reconSpace.matrixSize.z = 1
                    
                    encoding.reconSpace.fieldOfView_mm = ismrmrd.xsd.fieldOfViewMm()
                    encoding.reconSpace.fieldOfView_mm.x = float(num_x * voxel_size[0])
                    encoding.reconSpace.fieldOfView_mm.y = float(num_y * voxel_size[1])
                    encoding.reconSpace.fieldOfView_mm.z = float(voxel_size[2])
                    
                    encoding.encodingLimits = ismrmrd.xsd.encodingLimitsType()
                    mrdHead.encoding.append(encoding)
                    
                    mrdHead.sequenceParameters = ismrmrd.xsd.sequenceParametersType()
                    mrdHead.sequenceParameters.TR = [metadata.get('RepetitionTime', 1.0)]
                    mrdHead.sequenceParameters.TE = [metadata.get('EchoTime', 1.0)]
                    
                    mrdDset.write_xml_header(mrdHead.toXML('utf-8'))
                    print("✅ Written XML header")
                
                tmpMeta = ismrmrd.Meta()
                for key, value in metadata.items():
                    if isinstance(value, (list, tuple)):
                        tmpMeta[key] = list(value)
                    else:
                        tmpMeta[key] = str(value)
                
                tmpMeta['DataRole'] = 'Image'
                tmpMeta['ImageProcessingHistory'] = ['NIFTI_CONVERSION']
                tmpMeta['Keep_image_geometry'] = 1
                
                ismrmrd_image.attribute_string = tmpMeta.serialize()
                ismrmrd_image.image_series_index = metadata.get('SeriesNumber', 1)
                
                print(f"💾 Writing {num_z} slices x {num_c} channels as separate images...")
                
                image_counter = 1
                total_images = num_z * num_c
                
                # Double loops: write individual slice for each channel
                for slice_idx in range(num_z):
                    for channel_idx in range(num_c):
                        # Extract 2D slice specifically mapped to the channel
                        slice_data = ismrmrd_data[:, :, slice_idx, channel_idx].T.astype(np.float32)
                        
                        try:
                            slice_image = ismrmrd.Image.from_array(slice_data, transpose=False)
                        except TypeError:
                            slice_image = ismrmrd.Image.from_array(slice_data)
                        
                        slice_image.image_type = determined_image_type
                        slice_image.image_series_index = metadata.get('SeriesNumber', 1)
                        slice_image.image_index = image_counter
                        
                        slice_position = (
                            np.array(orientation_info['position']) + 
                            slice_idx * voxel_size[2] * np.array(orientation_info['slice_dir'])
                        )
                        
                        if hasattr(slice_image, 'position'):
                            slice_image.position[:] = slice_position.tolist()
                        if hasattr(slice_image, 'read_dir'):
                            slice_image.read_dir[:] = orientation_info['read_dir']
                        if hasattr(slice_image, 'phase_dir'):
                            slice_image.phase_dir[:] = orientation_info['phase_dir']
                        if hasattr(slice_image, 'slice_dir'):
                            slice_image.slice_dir[:] = orientation_info['slice_dir']
                        
                        slice_fov = [
                            num_x * voxel_size[0],
                            num_y * voxel_size[1],
                            voxel_size[2]
                        ]
                        
                        if hasattr(slice_image, 'field_of_view'):
                            slice_image.field_of_view[:] = slice_fov
                        
                        slice_meta = ismrmrd.Meta()
                        for key, value in metadata.items():
                            if isinstance(value, (list, tuple)):
                                slice_meta[key] = list(value)
                            else:
                                slice_meta[key] = str(value)
                        
                        slice_meta['DataRole'] = 'Image'
                        slice_meta['ImageProcessingHistory'] = ['NIFTI_CONVERSION']
                        slice_meta['Keep_image_geometry'] = 1
                        slice_meta['slice_number'] = slice_idx
                        slice_meta['channel_id'] = channel_idx   # <--- Channel ID added to XML Metadata block
                        slice_meta['position'] = slice_position.tolist()
                        
                        slice_image.attribute_string = slice_meta.serialize()
                        
                        slice_image.slice = slice_idx
                        slice_image.phase = 0
                        slice_image.acquisition_time_stamp = 0

                        series_index = metadata.get('SeriesNumber', 1)
                        mrdDset.append_image("image_%d" % series_index, slice_image)
                        
                        if image_counter % 50 == 0 or image_counter == total_images:
                            print(f"   Written {image_counter}/{total_images} slice-channel images...")
                            
                        image_counter += 1
                
                mrdDset.close()
                print(f"✅ Saved {total_images} images (Slices: {num_z}, Channels: {num_c}) to {output_path}")
            else:
                raise ImportError("ismrmrd.Dataset not available - cannot create proper ISMRMRD file")
            
        except (ImportError, AttributeError, Exception) as e:
            print(f"❌ Could not save as ISMRMRD file: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    return ismrmrd_image, metadata


def main():
    """Main function to test the converter"""
    print("🧪 NIfTI to ISMRMRD Converter (Multi-Channel & JSON Support)")
    print("=" * 50)
    
    parser = argparse.ArgumentParser(description="Convert a NIfTI file to ISMRMRD format for OpenRecon testing")
    parser.add_argument(
        "-i", "--input",
        dest="nifti_file",
        help="Path to the NIfTI file to convert",
        default="test.nii.gz"
    )
    parser.add_argument(
        "-o", "--output",
        dest="output_path",
        help="Optional output path for serialized ISMRMRD data",
        default="test_ismrmrd_output.h5"
    )
    args = parser.parse_args()

    nifti_file = args.nifti_file
    output_path = args.output_path
    
    if not os.path.exists(nifti_file):
        print(f"❌ Test file not found: {nifti_file}")
        print("   Please check the file path")
        return False
    
    try:
        print(f"➡️  Using input: {nifti_file}")
        print(f"➡️  Output path: {output_path}")
        ismrmrd_image, metadata = convert_nifti_to_ismrmrd(nifti_file, output_path)
        
        print("\n📋 Conversion Summary:")
        print(f"   Input file: {nifti_file}")
        print(f"   Original 4D volume shape: {ismrmrd_image.data.shape}")
        print(f"   Series: {metadata.get('SeriesNumber', 'Unknown')}")
        print(f"   PixelSpacing: {metadata.get('PixelSpacing', 'Unknown')}")
        print(f"   SliceThickness: {metadata.get('SliceThickness', 'Unknown')}")
        print(f"\n🧭 Orientation Information:")
        print(f"   First slice position: {metadata.get('position', 'Unknown')}")
        print(f"   Read direction: {metadata.get('read_dir', 'Unknown')}")
        print(f"   Phase direction: {metadata.get('phase_dir', 'Unknown')}")
        print(f"   Slice direction: {metadata.get('slice_dir', 'Unknown')}")
        print(f"   Slice spacing: {metadata.get('SliceThickness', 'Unknown')} mm")
        
        return ismrmrd_image, metadata
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = main()
    if result:
        print("\n🎉 Conversion completed successfully!")
    else:
        print("\n💥 Conversion failed!")