#!/bin/bash -f
# ****************************************************************************
# Vivado (TM) v2019.2 (64-bit)
#
# Filename    : compile.sh
# Simulator   : Mentor Graphics ModelSim Simulator
# Description : Script for compiling the simulation design source files
#
# Generated by Vivado on Fri Jun 09 19:59:01 BST 2023
# SW Build 2708876 on Wed Nov  6 21:39:14 MST 2019
#
# Copyright 1986-2019 Xilinx, Inc. All Rights Reserved.
#
# usage: compile.sh
#
# ****************************************************************************
bin_path="/mnt/applications/mentor/modelsim-2019.2/modelsim/modeltech/linux_x86_64"
set -Eeuo pipefail
source $FYP_DIR/scripts/add_hash.sh
source top_tb_compile.do > compile.log 2>&1

