export CUDA_VISIBLE_DEVICES=0 #,1,2,3,4,5,6,7
export HYDRA_FULL_ERROR=1

###################################
# User Configuration Section
###################################
# Set environment variables
export NUPLAN_DEVKIT_ROOT="/home/yzy/project/nuplan-devkit"  # nuplan-devkit absolute path (e.g., "/home/user/nuplan-devkit")
export NUPLAN_DATA_ROOT="/media/yzy/1E585F67585F3CA9/data_yzy"  # nuplan dataset absolute path (e.g. "/data")
export NUPLAN_MAPS_ROOT="/media/yzy/1E585F67585F3CA9/data_yzy/nuplan-v1.1/maps" # nuplan maps absolute path (e.g. "/data/nuplan-v1.1/maps")
export NUPLAN_EXP_ROOT="//media/yzy/1E585F67585F3CA9/data_yzy/nuplan-v1.1/exp" # nuplan experiment absolute path (e.g. "/data/nuplan-v1.1/exp")

# Dataset split to use
# Options: 
#   - "test14-random"
#   - "test14-hard"
#   - "val14"
SPLIT="mini"  # e.g., "val14"

# Challenge type
# Options: 
#   - "closed_loop_nonreactive_agents"
#   - "closed_loop_reactive_agents"
CHALLENGE="closed_loop_nonreactive_agents"  # e.g., "closed_loop_nonreactive_agents"
###################################


BRANCH_NAME=diffusion_planner_release
# ARGS_FILE=./checkpoints/args.json
# CKPT_FILE=./checkpoints/model.pth
ARGS_FILE=/home/yzy/project/Diffusion-Planner/checkpoints/args.json
CKPT_FILE=/home/yzy/project/Diffusion-Planner/checkpoints/model.pth

if [ "$SPLIT" == "val14" ]; then
    SCENARIO_BUILDER="nuplan"
else
    SCENARIO_BUILDER="nuplan_challenge"
fi
echo "Processing $CKPT_FILE..."
FILENAME=$(basename "$CKPT_FILE")
FILENAME_WITHOUT_EXTENSION="${FILENAME%.*}"

PLANNER=diffusion_planner

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=$CHALLENGE \
    planner=$PLANNER \
    planner.diffusion_planner.config.args_file=$ARGS_FILE \
    planner.diffusion_planner.ckpt_path=$CKPT_FILE \
    scenario_builder=$SCENARIO_BUILDER \
    scenario_filter=$SPLIT \
    experiment_uid=$PLANNER/$SPLIT/$BRANCH_NAME/${FILENAME_WITHOUT_EXTENSION}_$(date "+%Y-%m-%d-%H-%M-%S") \
    verbose=true \
    worker=ray_distributed \
    worker.threads_per_node=8 \
    distributed_mode='SINGLE_NODE' \
    number_of_gpus_allocated_per_simulation=0.1 5 \
    enable_simulation_progress_bar=true \
    hydra.searchpath="[pkg://diffusion_planner.config.scenario_filter, pkg://diffusion_planner.config, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments  ]"