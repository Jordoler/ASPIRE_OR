using MriResearchTools

function combine_coils_in_memory(mag5d, phase5d, TEsList)
    # juliacall passes python numpy arrays as Jjulia arrays
    combined = mcpc3ds(phase5d, mag5d, TEs=TEsList)
    
    # returns the combined 4D magnitude and phase arrays back to Python
    return combined.mag, combined.phase
end