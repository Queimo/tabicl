#!/usr/local_rwth/bin/zsh
### Job name
#SBATCH --job-name=LLEGNN
### File / path where STDOUT will be written, %J is the job id
#SBATCH --output=optimization-out_GEGNN.%J
#SBATCH --time=8:00:00
### Request memory you need for your job in MB
### SBATCH --mem-per-cpu=16000

### Request number of CPUs
#SBATCH --ntasks-per-node=8

### Change to the work directory
cd $HPCWORK/tabicl/
source $HOME/.zshrc
micromamba activate lle

python "$@"
