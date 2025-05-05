components = ["11", "loop", "ct"]
ells = [0, 2, 4]
folder_output = "/farmdisk1/mbonici/trained_effort_pybird_mnulcdm_2000"
for ell in ells
    for component in components
        # Construct the bsub command
        bsub_command = `bsub -P c7 -q medium -o /home/mbonici/emulator-zoo/Effort.jl/pybird_mnulcdm/job.out \
                            -e /home/mbonici/emulator-zoo/Effort.jl/pybird_mnulcdm/job.err -n 8 -M 12000 \
                            -R'span[hosts=1] select[hname!=teo22 && hname!=infne01 && hname!=totem04 && hname!=totem07 && hname!=totem08 && hname!=geant15 && hname!=geant16 && hname!=aiace12 && hname!=aiace13 && hname!=aiace14 && hname!=aiace15 && hname!=aiace16 && hname!=aiace17]' \
                            /home/mbonici/julia-1.10.7/bin/julia -t 8 \
                            /home/mbonici/emulator-zoo/Effort.jl/pybird_mnulcdm/trainer.jl \
                            --component $component -l $ell \
                            -i /farmdisk1/mbonici/effort_pybird_mnulcdm_2000 \
                            -o $folder_output
                            -p AsDzprec`

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
