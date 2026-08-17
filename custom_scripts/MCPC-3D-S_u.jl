using ArgParse
using MriResearchTools

# Usage: julia MCPC-3D-S.jl --mag_file <FILEPATH> --phase_file <FILEPATH> --TEsList TE0 TE1 TE2 ...

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--mag_file"
            help = "Path to the 5d mag file"
            arg_type = String
            required = true
        "--phase_file"
            help = "Path to the 5d phase file"
            arg_type = String
            required = true
        "--TEsList"
            help = "list of TEs"
            arg_type = Float64  # each echo tim to float64
            nargs = '+'
            required = true
    end

    return parse_args(s)
end

function combine_coils()
    parsed_args = parse_commandline()
    
    # Access the arguments using dictionary syntax
    mag5d_file = parsed_args["mag_file"]
    println("mag5d file: ", mag5d_file)
    phase5d_file = parsed_args["phase_file"]
    println("phase5d file: ", phase5d_file)
    TEsList = parsed_args["TEsList"]
    println("TEs List: ", TEsList)

    println("Loading 5D NIfTI files...")

    mag5d = readmag(mag5d_file)
    phase5d = readphase(phase5d_file)

    println("Performing MCPC-3D-S phase combination...")
     # 3. Perform MCPC-3D-S phase combination
    # combines channels to and outputs a 4D phase array (X, Y, Z, Echoes)
    combined = mcpc3ds(phase5d, mag5d, TEs=TEsList)
    println("MCPC-3D-S phase combination complete.")
    
    println("Saving combined 4D NIfTI files...")
    savenii(combined.phase, "combined_ph_4d.nii") #(X, Y, Z, Echoes)
    savenii(combined.mag, "combined_mag_4d.nii") #(X, Y, Z, Echoes)

    println("Processing complete! Files saved as combined_ph4d.nii and combined_mag4d.nii.")
    
    


end

combine_coils()