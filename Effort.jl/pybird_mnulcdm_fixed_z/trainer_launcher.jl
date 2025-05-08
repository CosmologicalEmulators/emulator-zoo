components = ["11", "loop", "ct"]
ells = [0, 2, 4]
iterations = [string(i) for i in 1:10]
preconditioning = "AsDzprec"
n_input = "20000"
input = "/farmdisk1/mbonici/effort_pybird_mnulcdm_fixed_z_" * n_input

for iteration in iterations
    folder_output = "/farmdisk1/mbonici/trained_effort_pybird_mnulcdm_fixed_z/trained_pybird_mnulcdm_fixed_z_" * n_input * "/" * preconditioning * "/" * iteration
    for ell in ells
        for component in components
            # Construct the bsub command
            bsub_command = `bsub -P c7 -q medium -o /home/mbonici/emulator-zoo/Effort.jl/pybird_mnulcdm_fixed_z/job.out \
                                -e /home/mbonici/emulator-zoo/Effort.jl/pybird_mnulcdm_fixed_z/job.err -n 4 -M 12000 \
                                -R'span[hosts=1] select[hname!=teo22 && hname!=infne01 && hname!=totem04 && hname!=totem07 && hname!=totem08 && hname!=geant15 && hname!=geant16 && hname!=aiace12 && hname!=aiace13 && hname!=aiace14 && hname!=aiace15 && hname!=aiace16 && hname!=aiace17]' \
                                /home/mbonici/julia-1.10.7/bin/julia -t 4 \
                                /home/mbonici/emulator-zoo/Effort.jl/pybird_mnulcdm_fixed_z/trainer.jl \
                                --component $component -l $ell \
                                -i $input \
                                -o $folder_output
                                -p $preconditioning`

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
end
