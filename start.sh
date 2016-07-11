#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/student/home/geigedav/Dokumente/MProject/MacCormackFluid/ThirdParty/GLFW/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/student/home/geigedav/Dokumente/MProject/MacCormackFluid/ThirdParty/GLEW/lib64
runcommand="mpirun -x LD_LIBRARY_PATH -np 5 -hostfile ./n3027_hostfile ./Release/MultiProject"
echo $runcommand
echo
$runcommand
