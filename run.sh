#!/bin/bash

declare -a configs=("default")

declare -a scripts=("preprocess.py" "simulator.py" "plotting.py" "all three" "Abort")

# source ~/.bashrc
# conda deactivate
# conda activate tc_env
# conda env list

PS3='Please enter your choice: '
select opt in "${scripts[@]}"; do
  case $opt in
  ${scripts[0]} | ${scripts[1]} | ${scripts[2]})
    printf "Running $opt for configs:\n"
    printf '%s\n' "${configs[@]}"
    for cfg in ${configs[*]}; do
      python $opt -c $cfg
      wait
    done
    break
    ;;
  "all three")
    printf "Running the three scripts for configs:\n"
    printf '%s\n' "${configs[@]}"
    for cfg in ${configs[*]}; do
      python ${scripts[0]} -c $cfg
      wait
      python ${scripts[1]} -c $cfg
      wait
      python ${scripts[2]} -c $cfg
    done
    break
    ;;
  "Abort")
    break
    ;;
  *) echo "invalid option $REPLY" ;;
  esac
done
