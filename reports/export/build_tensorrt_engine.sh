#!/bin/bash
# Construction du moteur TensorRT optimisé FP16 sur GPU cible
trtexec --onnx=neural_fmm_mgnn.onnx \
        --saveEngine=neural_fmm_mgnn_fp16.engine \
        --fp16 \
        --minShapes=node_features:32x7,node_coords:32x3,edge_near:2x64,edge_far:2x32,wind_3d:3 \
        --optShapes=node_features:2048x7,node_coords:2048x3,edge_near:2x8192,edge_far:2x2048,wind_3d:3 \
        --maxShapes=node_features:16384x7,node_coords:16384x3,edge_near:2x65536,edge_far:2x16384,wind_3d:3
