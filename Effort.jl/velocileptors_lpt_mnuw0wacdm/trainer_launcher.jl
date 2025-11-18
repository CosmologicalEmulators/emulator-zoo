# Define the arrays for components and -s values
# Define the array to loop over
components = ["11", "loop", "ct"]
ells = [0, 2, 4]

# Loop through each component and submit the bsub command
folder_output = "/farmdisk1/mbonici/trained_effort_velocileptors_lpt_mnuw0wacdm_200000"
for ell in ells
    for component in components
        # Construct the bsub command
        bsub_command = `bsub -P c7 -q long -o /home/mbonici/emulator-zoo/Effort.jl/velocileptors_lpt_mnuw0wacdm/job.out \
                            -e /home/mbonici/emulator-zoo/Effort.jl/velocileptors_lpt_mnuw0wacdm/job.err -n 8 -M 8100 \
                            -R'span[hosts=1] select[hname!=teo22 && hname!=infne01 && hname!=totem04 && hname!=totem07 && hname!=totem08 && hname!=geant15 && hname!=geant16 && hname!=aiace12 && hname!=aiace13 && hname!=aiace14 && hname!=aiace15 && hname!=aiace16 && hname!=aiace17]' \
                            julia -t 8 --project=/home/mbonici/emulator-zoo/Effort.jl/velocileptors_lpt_mnuw0wacdm \
                            /home/mbonici/emulator-zoo/Effort.jl/velocileptors_lpt_mnuw0wacdm/trainer.jl \
                            --component $component -l $ell \
                            -i /farmdisk1/mbonici/effort_velocileptors_lpt_mnuw0wacdm_200000 \
                            -o $folder_output`

        # Print the command for debugging (optional)
        println("Submitting job with --component=$component --ell=$ell")
        # Run the command
        run(bsub_command)
    end
    dest = joinpath(folder_output * "/" * string(ell), "biascombination.py")  # constructs the full destination path nicely
    run(`cp biascombination.py $dest`)
    dest = joinpath(folder_output * "/" * string(ell), "biascombination.jl")
    run(`cp biascombination.jl $dest`)
    dest = joinpath(folder_output * "/" * string(ell), "jacbiascombination.py")  # constructs the full destination path nicely
    run(`cp jacbiascombination.py $dest`)
    dest = joinpath(folder_output * "/" * string(ell), "jacbiascombination.jl")
    run(`cp jacbiascombination.jl $dest`)
end
