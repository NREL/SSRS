#!/bin/bash

declare -a configs=("default" "spring_mnorth_morning" "spring_mnorth_afternoon" "spring_mnorth_afternoon_random" "spring_mnorth_afternoon_cubic" "spring_mnorth_afternoon_k1" "spring_mnorth_afternoon_k2" "spring_mnorth_afternoon_2000t" "spring_mnorth_afternoon_256" "spring_mnorth_evening" "fall_msouth_morning" "fall_msouth_afternoon" "fall_msouth_evening")

declare -a scripts=("extract_data.py" "compute_updrafts.py" "generate_plots.py" "all three" "submit_EAGLE.sh" "Quit")

source ~/.bashrc
conda deactivate
conda activate tc_env
conda env list

PS3='Please enter your choice: '
select opt in "${scripts[@]}"; do
  case $opt in
  ${scripts[0]} | ${scripts[1]} | ${scripts[2]})
    printf "Running $opt for configs:\n"
    printf '%s\n' "${configs[@]}"
    for cfg in ${configs[*]}; do
      python $opt $cfg
      wait
    done
    break
    ;;
  "all three")
    printf "Running the three scripts for configs:\n"
    printf '%s\n' "${configs[@]}"
    for cfg in ${configs[*]}; do
      python ${scripts[0]} $cfg
      wait
      python ${scripts[1]} $cfg
      wait
      python ${scripts[2]} $cfg
    done
    break
    ;;
  ${scripts[4]})
    printf "Submitting jobs to EAGLE HPC\n"
    #printf '%s\n' "${configs[@]}"
    for cfg in ${configs[*]}; do
      sbatch $opt $cfg
      wait
    done
    break
    ;;

  "Quit")
    break
    ;;
  *) echo "invalid option $REPLY" ;;
  esac
done
