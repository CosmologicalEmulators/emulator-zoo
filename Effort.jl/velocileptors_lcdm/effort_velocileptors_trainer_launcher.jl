# Define the arrays for components and -s values
# Define the array to loop over
components = ["11", "loop", "ct", "st"]
ells = [0, 2, 4]

# Loop through each component and submit the bsub command
folder_output = "/farmdisk1/mbonici/trained_effort_velocileptors_lcdm_1000"
for ell in ells
    for component in components
        # Construct the bsub command
        bsub_command = `bsub -P c7 -q medium -o /home/mbonici/emulator-zoo/Effort.jl/velocileptors_lcdm/job.out \
                            -e /home/mbonici/emulator-zoo/Effort.jl/velocileptors_lcdm/job.err -n 8 -M 6100 \
                            -R'span[hosts=1] select[hname!=teo22 && hname!=infne01 && hname!=totem04 && hname!=totem07 && hname!=totem08 && hname!=geant15 && hname!=geant16 && hname!=aiace12 && hname!=aiace13 && hname!=aiace14 && hname!=aiace15 && hname!=aiace16 && hname!=aiace17]' \
                            /home/mbonici/julia-1.9.1/bin/julia -t 8 \
                            /home/mbonici/emulator-zoo/Effort.jl/velocileptors_lcdm/effort_velocileptors_trainer.jl \
                            --component $component -l $ell \
                            -i /farmdisk1/mbonici/effort_velocileptors_lcdm_1000 \
                            -o $folder_output`

        # Print the command for debugging (optional)
        println("Submitting job with --component=$component --ell=$ell")
        # Run the command
        run(bsub_command)
    end
    dest = joinpath(folder_output * "/" * string(ell), "biascontraction.py")  # constructs the full destination path nicely
    run(`cp biascontraction.py $dest`)
    dest = joinpath(folder_output * "/" * string(ell), "biascontraction.jl")
    run(`cp biascontraction.jl $dest`)
end
