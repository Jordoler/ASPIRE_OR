import ismrmrd
import logging
import numpy as np
import custom_scripts.mcpc_sort as sort

class Sorter():
    def __init__(self,metadata,meta_templates:bool=True,type:str="i") -> None:
        """Initializes the sorter instance and allocates 5D arrays based on MRD header (metadata)"""
        if type=="i":
            number_of_coils = metadata.acquisitionSystemInformation.receiverChannels
            TEs = metadata.sequenceParameters.TE 
            slices = metadata.encoding[0].encodedSpace.matrixSize.z
            height = metadata.encoding[0].encodedSpace.matrixSize.y
            width = metadata.encoding[0].encodedSpace.matrixSize.x

            self.type = type
            self.meta_temps = meta_templates
            self.Necho = len(TEs)
            self.TEs = [float(t) for t in TEs]

            self.ph_filled = np.zeros((slices, self.Necho, number_of_coils), dtype=bool)
            self.mag_filled = np.zeros((slices, self.Necho, number_of_coils), dtype=bool)

            self.mag_data_5d = np.zeros((width, height, slices, self.Necho, number_of_coils), dtype=np.float32,order="F")
            self.ph_data_5d  = np.zeros((width, height, slices, self.Necho, number_of_coils), dtype=np.float32,order="F")

            if meta_templates:
                self.mag_meta_templates = np.empty((self.Necho, slices), dtype=object)
                self.ph_meta_templates = np.empty((self.Necho, slices), dtype=object)
        if type=="a":
            ## TODO
            logging.error("Sorter - Error: Sorted does not yet handle acquisition data")


    def collect_image(self,item) -> None:
        if isinstance(item,ismrmrd.Image): #check not necesssary
            echo_number, slice_no = sort.get_EchoNo_and_SliceNo(item)
            coil = sort.get_coil(item)
            meta = ismrmrd.Meta.deserialize(item.attribute_string)
            echo_id = int(echo_number) - 1
            coil_id = int(coil) - 1
            slice_id = int(slice_no) - 1

            raw_data_2d = item.data[0, 0, :, :].T 

            if item.image_type is ismrmrd.IMTYPE_PHASE:
                if not self.ph_filled[slice_id, echo_id, coil_id]:
                    self.ph_data_5d[:, :, slice_id, echo_id, coil_id] = raw_data_2d
                    self.ph_filled[slice_id, echo_id, coil_id] = True

                if coil_id == 0 and self.meta_temps:
                    self.ph_meta_templates[echo_id, slice_id] = {'head': item.getHead(),
                                                                'meta': meta}#ismrmrd.Meta.deserialize(item.attribute_string)}

            elif item.image_type is ismrmrd.IMTYPE_MAGNITUDE:
                if not self.mag_filled[slice_id, echo_id, coil_id]:
                    self.mag_data_5d[:, :, slice_id, echo_id, coil_id] = raw_data_2d
                    self.mag_filled[slice_id, echo_id, coil_id] = True
                if coil_id == 0 and self.meta_temps:
                    self.mag_meta_templates[echo_id, slice_id] = {'head': item.getHead(),
                                                                'meta': meta}#ismrmrd.Meta.deserialize(item.attribute_string)}
        return None

    def is_phase_full(self) -> bool:
        """Returns True if all phase slices, echoes, and coils have been collected."""
        return bool(self.ph_filled.all())

    def is_mag_full(self) -> bool:
        """Returns True if all magnitude slices, echoes, and coils have been collected."""
        return bool(self.mag_filled.all())

    def is_complete(self) -> bool:
        """Returns True if both phase and magnitude data are fully collected."""
        return self.is_phase_full() and self.is_mag_full()


    def get_phase_shape(self) -> tuple:
            """Returns 5D spatial/channel shape: (width, height, slices, Necho, coils)."""
            return self.ph_data_5d.shape

    def get_mag_shape(self) -> tuple:
        """Returns 5D spatial/channel shape: (width, height, slices, Necho, coils)."""
        return self.mag_data_5d.shape

    def get_phase_echo(self, echo_number: int) -> np.ndarray:
        """Extracts 4D phase data (width, height, slices, coils) for a 1-indexed echo."""
        echo_id = echo_number - 1
        if not (0 <= echo_id < self.Necho):
            raise IndexError(f"Echo number {echo_number} out of bounds (1 to {self.Necho}).")
        return self.ph_data_5d[:, :, :, echo_id, :]

    def get_mag_echo(self, echo_number: int) -> np.ndarray:
        """Extracts 4D magnitude data (width, height, slices, coils) for a 1-indexed echo."""
        echo_id = echo_number - 1
        if not (0 <= echo_id < self.Necho):
            raise IndexError(f"Echo number {echo_number} out of bounds (1 to {self.Necho}).")
        return self.mag_data_5d[:, :, :, echo_id, :]

    def get_filled_channels(self) -> dict:
        """
        Returns lists of 0-indexed coil/channel IDs that have collected at least one slice/echo.
        """
        # ph_filled shape: (slices, Necho, number_of_coils)
        ph_channels = np.where(self.ph_filled.any(axis=(0, 1)))[0].tolist()
        mag_channels = np.where(self.mag_filled.any(axis=(0, 1)))[0].tolist()
        return {
            "phase": ph_channels,
            "magnitude": mag_channels
        }
