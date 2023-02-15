# RUN Docker ([ref](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/326_YOLOPv2/post_process_gen_tools/convert_script.txt))

# Convert YOLOPv2 to ONNX with NMS for batch = 1


OPSET=11
WIDTH=640
HEIGHT=384
BATCHES=1
CLASSES=80

python3 utils/set_boxes_var.py --width ${WIDTH} --height ${HEIGHT} > /tmp/PARAM_LIST
PARAM_LIST=$(</tmp/PARAM_LIST)
HWBOX=(`echo ${PARAM_LIST[i]}`)
BOXES=${HWBOX[0]}
BOXES1=${HWBOX[0]}
BOXES2=${HWBOX[1]}
BOXES3=${HWBOX[2]}


# Convert torch pt to ONNX without NMS and Split for trace

python3 utils/torch2onnx.py


# Remove Nodes and Rename ops

snd4onnx \
--remove_node_names SequenceConstruct_470 \
--input_onnx_file_path data/graphs/yolopv2_${HEIGHT}x${WIDTH}.onnx \
--output_onnx_file_path data/graphs/yolopv2_${HEIGHT}x${WIDTH}.onnx
onnxsim data/graphs/yolopv2_${HEIGHT}x${WIDTH}.onnx data/graphs/yolopv2_${HEIGHT}x${WIDTH}.onnx
onnxsim data/graphs/yolopv2_${HEIGHT}x${WIDTH}.onnx data/graphs/yolopv2_${HEIGHT}x${WIDTH}.onnx
onnxsim data/graphs/yolopv2_${HEIGHT}x${WIDTH}.onnx data/graphs/yolopv2_${HEIGHT}x${WIDTH}.onnx
sor4onnx \
--input_onnx_file_path data/graphs/yolopv2_${HEIGHT}x${WIDTH}.onnx \
--old_new "onnx::SequenceConstruct_599" "pred0" \
--mode outputs \
--output_onnx_file_path data/graphs/yolopv2_${HEIGHT}x${WIDTH}.onnx

sor4onnx \
--input_onnx_file_path data/graphs/yolopv2_${HEIGHT}x${WIDTH}.onnx \
--old_new "onnx::SequenceConstruct_602" "pred1" \
--mode outputs \
--output_onnx_file_path data/graphs/yolopv2_${HEIGHT}x${WIDTH}.onnx

sor4onnx \
--input_onnx_file_path data/graphs/yolopv2_${HEIGHT}x${WIDTH}.onnx \
--old_new "onnx::SequenceConstruct_605" "pred2" \
--mode outputs \
--output_onnx_file_path data/graphs/yolopv2_${HEIGHT}x${WIDTH}.onnx



# Remove anchor grid

echo ====== processing yolopv2_${HEIGHT}x${WIDTH}  ====== 



snd4onnx \
    --remove_node_names anchor_grid0 \
    --input_onnx_file_path data/graphs/yolopv2_${HEIGHT}x${WIDTH}.onnx \
    --output_onnx_file_path data/graphs/yolopv2_${HEIGHT}x${WIDTH}.onnx

snd4onnx \
    --remove_node_names anchor_grid1 \
    --input_onnx_file_path data/graphs/yolopv2_${HEIGHT}x${WIDTH}.onnx \
    --output_onnx_file_path data/graphs/yolopv2_${HEIGHT}x${WIDTH}.onnx

snd4onnx \
    --remove_node_names anchor_grid2 \
    --input_onnx_file_path data/graphs/yolopv2_${HEIGHT}x${WIDTH}.onnx \
    --output_onnx_file_path data/graphs/yolopv2_${HEIGHT}x${WIDTH}.onnx


# rename shapes

sio4onnx \
--input_onnx_file_path data/graphs/yolopv2_${HEIGHT}x${WIDTH}.onnx \
--output_onnx_file_path data/graphs/yolopv2_${HEIGHT}x${WIDTH}.onnx \
--input_names "input" \
--input_shapes 1 3 "height" "width" \
--output_names "seg" \
--output_names "ll" \
--output_names "pred0" \
--output_names "pred1" \
--output_names "pred2" \
--output_shapes 1 2 "height" "width" \
--output_shapes 1 1 "height" "width" \
--output_shapes 1 255 "pred0h" "pred0w" \
--output_shapes 1 255 "pred1h" "pred1w" \
--output_shapes 1 255 "pred2h" "pred2w"


# Add batch

sbi4onnx \
--input_onnx_file_path data/graphs/yolopv2_${HEIGHT}x${WIDTH}.onnx \
--output_onnx_file_path data/graphs/yolopv2_Nx3x${HEIGHT}x${WIDTH}.onnx \
--initialization_character_string batch

# Split for trace  ONNX

python3 utils/make_split_for_trace_model.py --width ${WIDTH} --height ${HEIGHT}


It will generate onnx-file in graphs dir.

# Make Boxes Scores ONNX

python3 utils/make_boxes_scores.py --width ${WIDTH} --height ${HEIGHT} --anchors-num 3 --batches ${BATCHES} -o ${OPSET} -c ${CLASSES}


# Make CXCYWH to Y1X1Y2X2 conv ONNX

python3 utils/make_cxcywh_y1x1y2x2.py --width ${WIDTH} --height ${HEIGHT} --anchors-num 3 --batches ${BATCHES} -o ${OPSET}


# combine(merge) onnx models [Boxes Scores + cxcywh_y1x1y2x2]

snc4onnx \
    --input_onnx_file_paths data/graphs/boxes_scores.onnx data/graphs/cxcywh_y1x1y2x2.onnx \
    --srcop_destop boxes_cxcywh cxcywh \
    --output_onnx_file_path data/graphs/boxes_y1x1y2x2_scores.onnx


# NMS

## Generate constant - maximum number of boxes per class (default 20)


sog4onnx \
    --op_type Constant \
    --opset ${OPSET} \
    --op_name max_output_boxes_per_class_const \
    --output_variables max_output_boxes_per_class int64 [1] \
    --attributes value int64 [20] \
    --output_onnx_file_path data/graphs/Constant_max_output_boxes_per_class.onnx


## Generate constant - iou_threshold (default 0.5)

sog4onnx \
    --op_type Constant \
    --opset ${OPSET} \
    --op_name iou_threshold_const \
    --output_variables iou_threshold float32 [1] \
    --attributes value float32 [0.5] \
    --output_onnx_file_path data/graphs/Constant_iou_threshold.onnx


## Generate constant - score_threshold (default 0.3)

sog4onnx \
    --op_type Constant \
    --opset ${OPSET} \
    --op_name score_threshold_const \
    --output_variables score_threshold float32 [1] \
    --attributes value float32 [0.3] \
    --output_onnx_file_path data/graphs/Constant_score_threshold.onnx


## Generate NMS Operation 

OP=NonMaxSuppression
LOWEROP=${OP,,}
sog4onnx \
--op_type ${OP} \
--opset ${OPSET} \
--op_name ${LOWEROP}${OPSET} \
--input_variables boxes_var float32 [${BATCHES},${BOXES},4] \
--input_variables scores_var float32 [${BATCHES},${CLASSES},${BOXES}] \
--input_variables max_output_boxes_per_class_var int64 [1] \
--input_variables iou_threshold_var float32 [1] \
--input_variables score_threshold_var float32 [1] \
--output_variables selected_indices int64 [\'N\',3] \
--attributes center_point_box int64 0 \
--output_onnx_file_path data/graphs/${OP}${OPSET}.onnx


## Combine Constants with NMS Operation

snc4onnx \
    --input_onnx_file_paths data/graphs/Constant_max_output_boxes_per_class.onnx data/graphs/NonMaxSuppression11.onnx \
    --srcop_destop max_output_boxes_per_class max_output_boxes_per_class_var \
    --output_onnx_file_path data/graphs/NonMaxSuppression${OPSET}.onnx

snc4onnx \
    --input_onnx_file_paths data/graphs/Constant_iou_threshold.onnx data/graphs/NonMaxSuppression${OPSET}.onnx \
    --srcop_destop iou_threshold iou_threshold_var \
    --output_onnx_file_path data/graphs/NonMaxSuppression${OPSET}.onnx

snc4onnx \
    --input_onnx_file_paths data/graphs/Constant_score_threshold.onnx data/graphs/NonMaxSuppression${OPSET}.onnx \
    --srcop_destop score_threshold score_threshold_var \
    --output_onnx_file_path data/graphs/NonMaxSuppression${OPSET}.onnx


## Force a change in the opsen of an ONNX graph (if needed)

soc4onnx \
    --input_onnx_file_path data/graphs/NonMaxSuppression${OPSET}.onnx \
    --output_onnx_file_path data/graphs/NonMaxSuppression${OPSET}.onnx \
    --opset ${OPSET}


# Combine Boxes + Scores + NMS

snc4onnx \
    --input_onnx_file_paths data/graphs/boxes_y1x1y2x2_scores.onnx data/graphs/NonMaxSuppression${OPSET}.onnx \
    --srcop_destop scores scores_var y1x1y2x2 boxes_var \
    --output_onnx_file_path data/graphs/nms_yolopv2.onnx


# Myriad workarount Mul Operation Create

OP=Mul
LOWEROP=${OP,,}
OPSET=${OPSET}
sog4onnx \
--op_type ${OP} \
--opset ${OPSET} \
--op_name ${LOWEROP}${OPSET} \
--input_variables workaround_mul_a int64 [\'N\',3] \
--input_variables workaround_mul_b int64 [1] \
--output_variables workaround_mul_out int64 [\'N\',3] \
--output_onnx_file_path data/graphs/${OP}${OPSET}_workaround.onnx


## Myriad workaround Constant

sog4onnx \
    --op_type Constant \
    --opset ${OPSET} \
    --op_name workaround_mul_const_op \
    --output_variables workaround_mul_const int64 [1] \
    --attributes value int64 [1] \
    --output_onnx_file_path data/graphs/Constant_workaround_mul.onnx


## Combine Mul operation + Constant

snc4onnx \
    --input_onnx_file_paths data/graphs/Constant_workaround_mul.onnx data/graphs/Mul${OPSET}_workaround.onnx \
    --srcop_destop workaround_mul_const workaround_mul_b \
    --output_onnx_file_path data/graphs/Mul${OPSET}_workaround.onnx


# Combine NMS + Myriad workaround

snc4onnx \
    --input_onnx_file_paths data/graphs/nms_yolopv2.onnx data/graphs/Mul${OPSET}_workaround.onnx \
    --srcop_destop selected_indices workaround_mul_a \
    --output_onnx_file_path data/graphs/nms_yolopv2.onnx


# Clean

rm data/graphs/boxes_scores.onnx
rm data/graphs/Constant_iou_threshold.onnx
rm data/graphs/Constant_max_output_boxes_per_class.onnx
rm data/graphs/Constant_score_threshold.onnx
rm data/graphs/Constant_workaround_mul.onnx
rm data/graphs/cxcywh_y1x1y2x2.onnx
rm data/graphs/Mul${OPSET}_workaround.onnx
rm data/graphs/NonMaxSuppression${OPSET}.onnx
rm data/graphs/boxes_y1x1y2x2_scores.onnx



# Score GatherND Operation ONNX Creation
## [gather_nd()](https://www.geeksforgeeks.org/python-tensorflow-gather_nd/) is used to gather the slice from input tensor based on the indices provided.


python3 utils/make_score_gather_nd.py -b ${BATCHES} -x ${BOXES} -c ${CLASSES}

We use tf here because there is no analog of gather_nd operation in torch still


## Conver to ONNX

python -m tf2onnx.convert \
    --opset ${OPSET} \
    --tflite data/saved_model_postprocess/nms_score_gather_nd.tflite \
    --output data/graphs/nms_score_gather_nd.onnx


## Rename Operations names

### remove ":0"s from names after tf to onnx conv

sor4onnx \
    --input_onnx_file_path data/graphs/nms_score_gather_nd.onnx \
    --old_new ":0" "" \
    --output_onnx_file_path data/graphs/nms_score_gather_nd.onnx


### Rename all

sor4onnx \
    --input_onnx_file_path data/graphs/nms_score_gather_nd.onnx \
    --old_new "serving_default_input_1" "gn_scores" \
    --output_onnx_file_path data/graphs/nms_score_gather_nd.onnx \
    --mode inputs



sor4onnx \
    --input_onnx_file_path data/graphs/nms_score_gather_nd.onnx \
    --old_new "serving_default_input_2" "gn_selected_indices" \
    --output_onnx_file_path data/graphs/nms_score_gather_nd.onnx \
    --mode inputs



sor4onnx \
    --input_onnx_file_path data/graphs/nms_score_gather_nd.onnx \
    --old_new "PartitionedCall" "final_scores" \
    --output_onnx_file_path data/graphs/nms_score_gather_nd.onnx \
    --mode outputs


## Update shape

python3 utils/make_input_output_shape_update.py \
    --input_onnx_file_path data/graphs/nms_score_gather_nd.onnx \
    --output_onnx_file_path data/graphs/nms_score_gather_nd.onnx \
    --input_names gn_scores \
    --input_names gn_selected_indices \
    --input_shapes ${BATCHES} ${CLASSES} ${BOXES} \
    --input_shapes N 3 \
    --output_names final_scores \
    --output_shapes N 1


## Simplify

onnxsim data/graphs/nms_score_gather_nd.onnx data/graphs/nms_score_gather_nd.onnx


# Combine NMS + Score GatherND

snc4onnx \
    --input_onnx_file_paths data/graphs/nms_yolopv2.onnx data/graphs/nms_score_gather_nd.onnx \
    --srcop_destop class_scores gn_scores workaround_mul_out gn_selected_indices \
    --output_onnx_file_path data/graphs/nms_yolopv2_nd.onnx


## Simplify

onnxsim data/graphs/nms_yolopv2_nd.onnx data/graphs/nms_yolopv2_nd.onnx


# Final Batch Nums ONNX

python3 utils/make_final_batch_nums_final_class_nums_final_box_nums.py


# Boxes Gather ND ONNX

python3 utils/make_box_gather_nd.py


## Convert tf to onnx

python3 -m tf2onnx.convert \
    --opset ${OPSET} \
    --tflite data/saved_model_postprocess/nms_box_gather_nd.tflite \
    --output data/graphs/nms_box_gather_nd.onnx


## Remove ":0"s after conv

sor4onnx \
    --input_onnx_file_path data/graphs/nms_box_gather_nd.onnx \
    --old_new ":0" "" \
    --output_onnx_file_path data/graphs/nms_box_gather_nd.onnx


## Rename ops inputs and outputs

sor4onnx \
    --input_onnx_file_path data/graphs/nms_box_gather_nd.onnx \
    --old_new "serving_default_input_1" "gn_boxes" \
    --output_onnx_file_path data/graphs/nms_box_gather_nd.onnx \
    --mode inputs



sor4onnx \
    --input_onnx_file_path data/graphs/nms_box_gather_nd.onnx \
    --old_new "serving_default_input_2" "gn_selected_indices" \
    --output_onnx_file_path data/graphs/nms_box_gather_nd.onnx \
    --mode inputs



sor4onnx \
    --input_onnx_file_path data/graphs/nms_box_gather_nd.onnx \
    --old_new "PartitionedCall" "final_boxes" \
    --output_onnx_file_path data/graphs/nms_box_gather_nd.onnx \
    --mode outputs


## Update shapes

python3 utils/make_input_output_shape_update.py \
    --input_onnx_file_path data/graphs/nms_box_gather_nd.onnx \
    --output_onnx_file_path data/graphs/nms_box_gather_nd.onnx \
    --input_names gn_boxes \
    --input_names gn_selected_indices \
    --input_shapes ${BATCHES} ${BOXES} 4 \
    --input_shapes N 2 \
    --output_names final_boxes \
    --output_shapes N 4


## Simplify ONNX

onnxsim data/graphs/nms_box_gather_nd.onnx data/graphs/nms_box_gather_nd.onnx


## Clean

rm data/graphs/nms_score_gather_nd.onnx
rm data/graphs/nms_yolopv2.onnx


# NMS_YOLOPV2_ND + NMS_FINAL_BATCH Combine

snc4onnx \
    --input_onnx_file_paths data/graphs/nms_yolopv2_nd.onnx data/graphs/nms_final_batch_nums_final_class_nums_final_box_nums.onnx \
    --srcop_destop workaround_mul_out bc_input \
    --op_prefixes_after_merging main01 sub01 \
    --output_onnx_file_path data/graphs/nms_yolopv2_split.onnx


# Combine NMS_SPLIT+NMS_BOX_GATHER_ND

snc4onnx \
    --input_onnx_file_paths data/graphs/nms_yolopv2_split.onnx data/graphs/nms_box_gather_nd.onnx \
    --srcop_destop main01_y1x1y2x2 gn_boxes final_box_nums gn_selected_indices \
    --output_onnx_file_path data/graphs/nms_yolopv2_merged.onnx

## Simplify 

onnxsim data/graphs/nms_yolopv2_merged.onnx data/graphs/nms_yolopv2_merged.onnx


# Clean NMS Operations names

sor4onnx \
    --input_onnx_file_path data/graphs/nms_yolopv2_merged.onnx \
    --old_new "main01_final_scores" "final_scores" \
    --output_onnx_file_path data/graphs/nms_yolopv2_merged.onnx \
    --mode outputs



sor4onnx \
    --input_onnx_file_path data/graphs/nms_yolopv2_merged.onnx \
    --old_new "sub01_final_batch_nums" "final_batch_nums" \
    --output_onnx_file_path data/graphs/nms_yolopv2_merged.onnx \
    --mode outputs



sor4onnx \
    --input_onnx_file_path data/graphs/nms_yolopv2_merged.onnx \
    --old_new "sub01_final_class_nums" "final_class_nums" \
    --output_onnx_file_path data/graphs/nms_yolopv2_merged.onnx \
    --mode outputs


# NMS Output merge

python3 utils/make_nms_outputs_merge.py


## Simplify

onnxsim data/graphs/nms_batchno_classid_y1x1y2x2_cat.onnx data/graphs/nms_batchno_classid_y1x1y2x2_cat.onnx


# Merge ONNXs

snc4onnx \
    --input_onnx_file_paths data/graphs/nms_yolopv2_merged.onnx data/graphs/nms_batchno_classid_y1x1y2x2_cat.onnx \
    --srcop_destop final_batch_nums cat_batch final_class_nums cat_classid final_boxes cat_y1x1y2x2 \
    --output_onnx_file_path data/graphs/nms_yolopv2.onnx


## Rename op

sor4onnx \
    --input_onnx_file_path data/graphs/nms_yolopv2.onnx \
    --old_new "final_scores" "score" \
    --output_onnx_file_path data/graphs/nms_yolopv2.onnx \
    --mode outputs


# Clean

rm data/graphs/nms_batchno_classid_y1x1y2x2_cat.onnx
rm data/graphs/nms_box_gather_nd.onnx
rm data/graphs/nms_final_batch_nums_final_class_nums_final_box_nums.onnx
rm data/graphs/nms_yolopv2_merged.onnx
rm data/graphs/nms_yolopv2_nd.onnx
rm data/graphs/nms_yolopv2_split.onnx


# Add Batches to NMS N

python3 utils/make_batch_initialize.py \
    -if data/graphs/nms_yolopv2.onnx \
    -of data/graphs/nms_yolopv2_N.onnx


# Split N Batch

echo ====== processing yolopv2_Nx3x${HEIGHT}x${WIDTH} ====== 



sbi4onnx \
    --input_onnx_file_path data/graphs/split_for_trace_model_${HEIGHT}x${WIDTH}.onnx \
    --output_onnx_file_path data/graphs/split_for_trace_model_Nx3x${HEIGHT}x${WIDTH}.onnx \
    --initialization_character_string batch



echo ====== processing yolopv2_Nx3x${HEIGHT}x${WIDTH}  ====== 

##  Rewrite parameters

sam4onnx \
    --input_onnx_file_path data/graphs/split_for_trace_model_Nx3x${HEIGHT}x${WIDTH}.onnx \
    --output_onnx_file_path data/graphs/split_for_trace_model_Nx3x${HEIGHT}x${WIDTH}.onnx \
    --op_name Reshape_78 \
    --input_constants onnx::Reshape_381 int64 [-1,${BOXES1},85]
sam4onnx \
    --input_onnx_file_path data/graphs/split_for_trace_model_Nx3x${HEIGHT}x${WIDTH}.onnx \
    --output_onnx_file_path data/graphs/split_for_trace_model_Nx3x${HEIGHT}x${WIDTH}.onnx \
    --op_name Reshape_150 \
    --input_constants onnx::Reshape_381 int64 [-1,${BOXES2},85]
sam4onnx \
    --input_onnx_file_path data/graphs/split_for_trace_model_Nx3x${HEIGHT}x${WIDTH}.onnx \
    --output_onnx_file_path data/graphs/split_for_trace_model_Nx3x${HEIGHT}x${WIDTH}.onnx \
    --op_name Reshape_222 \
    --input_constants onnx::Reshape_381 int64 [-1,${BOXES3},85]



# Combine YOLO + Post-Process

echo ====== processing yolopv2_${HEIGHT}x${WIDTH}  ====== 



snc4onnx \
    --input_onnx_file_paths data/graphs/yolopv2_${HEIGHT}x${WIDTH}.onnx data/graphs/split_for_trace_model_${HEIGHT}x${WIDTH}.onnx \
    --srcop_destop pred0 split_for_trace_model_pred0 pred1 split_for_trace_model_pred1 pred2 split_for_trace_model_pred2 \
    --op_prefixes_after_merging main split \
    --output_onnx_file_path data/graphs/yolopv2_split_${HEIGHT}x${WIDTH}.onnx
snc4onnx \
    --input_onnx_file_paths data/graphs/yolopv2_split_${HEIGHT}x${WIDTH}.onnx data/graphs/nms_yolopv2.onnx \
    --srcop_destop split_for_trace_model_pred predictions \
    --output_onnx_file_path data/graphs/yolopv2_post_${HEIGHT}x${WIDTH}.onnx
sor4onnx \
    --input_onnx_file_path data/graphs/yolopv2_post_${HEIGHT}x${WIDTH}.onnx \
    --old_new "main_" "" \
    --mode outputs \
    --output_onnx_file_path data/graphs/yolopv2_post_${HEIGHT}x${WIDTH}.onnx
sor4onnx \
    --input_onnx_file_path data/graphs/yolopv2_split_${HEIGHT}x${WIDTH}.onnx \
    --old_new "main_" "" \
    --mode outputs \
    --output_onnx_file_path data/graphs/yolopv2_split_${HEIGHT}x${WIDTH}.onnx

onnxsim data/graphs/yolopv2_split_${HEIGHT}x${WIDTH}.onnx data/graphs/yolopv2_split_${HEIGHT}x${WIDTH}.onnx
onnxsim data/graphs/yolopv2_split_${HEIGHT}x${WIDTH}.onnx data/graphs/yolopv2_split_${HEIGHT}x${WIDTH}.onnx



echo ====== processing yolopv2_Nx3x${HEIGHT}x${WIDTH}  ====== 



snc4onnx \
    --input_onnx_file_paths data/graphs/yolopv2_Nx3x${HEIGHT}x${WIDTH}.onnx data/graphs/split_for_trace_model_Nx3x${HEIGHT}x${WIDTH}.onnx \
    --srcop_destop pred0 split_for_trace_model_pred0 pred1 split_for_trace_model_pred1 pred2 split_for_trace_model_pred2 \
    --op_prefixes_after_merging main split \
    --output_onnx_file_path data/graphs/yolopv2_split_Nx3x${HEIGHT}x${WIDTH}.onnx
snc4onnx \
    --input_onnx_file_paths data/graphs/yolopv2_split_Nx3x${HEIGHT}x${WIDTH}.onnx data/graphs/nms_yolopv2_N.onnx \
    --srcop_destop split_for_trace_model_pred predictions \
    --output_onnx_file_path data/graphs/yolopv2_post_Nx3x${HEIGHT}x${WIDTH}.onnx
sor4onnx \
    --input_onnx_file_path data/graphs/yolopv2_post_Nx3x${HEIGHT}x${WIDTH}.onnx \
    --old_new "main_" "" \
    --mode outputs \
    --output_onnx_file_path data/graphs/yolopv2_post_Nx3x${HEIGHT}x${WIDTH}.onnx
sor4onnx \
    --input_onnx_file_path data/graphs/yolopv2_split_Nx3x${HEIGHT}x${WIDTH}.onnx \
    --old_new "main_" "" \
    --mode outputs \
    --output_onnx_file_path data/graphs/yolopv2_split_Nx3x${HEIGHT}x${WIDTH}.onnx
onnxsim data/graphs/yolopv2_split_Nx3x${HEIGHT}x${WIDTH}.onnx data/graphs/yolopv2_split_Nx3x${HEIGHT}x${WIDTH}.onnx
onnxsim data/graphs/yolopv2_split_Nx3x${HEIGHT}x${WIDTH}.onnx data/graphs/yolopv2_split_Nx3x${HEIGHT}x${WIDTH}.onnx
