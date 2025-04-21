components = ["11", "loop", "ct"]
ells = [0, 2, 4]
for ell in ells
    for component in components
        # Construct the bsub command
        bsub_command = `bsub -P c7 -q medium -o /home/mbonici/EmulatorsTrainer.jl/scripts/job.out \
                            -e /home/mbonici/EmulatorsTrainer.jl/scripts/job.err -n 8 -M 12000 \
                            -R'span[hosts=1] select[hname!=teo22 && hname!=infne01 && hname!=totem04 && hname!=totem07 && hname!=totem08 && hname!=geant15 && hname!=geant16 && hname!=aiace12 && hname!=aiace13 && hname!=aiace14 && hname!=aiace15 && hname!=aiace16 && hname!=aiace17]' \
                            /home/mbonici/julia-1.10.7/bin/julia -t 8 \
                            /home/mbonici/emulator-zoo/Effort.jl/mnuw0wacdm/effort_pybird_trainer_mnuw0wacdm.jl \
                            --component $component -l $ell \
                            -i /farmdisk1/mbonici/test_pybird_120000_mnuw0wacdm \
                            -o /farmdisk1/mbonici/batch_trained_pybird_120000_mnuw0wacdm_small_nn_AsDzprec
                            -p AsDzprec`

        # Print the command for debugging (optional)
        println("Submitting job with --component=$component --ell=$ell")

        # Run the command
        run(bsub_command)
    end
end
