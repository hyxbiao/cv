
attr_key=${1:-skirt_length_labels}
mode=${2:-train}
debug=$3

base_model_dir=models/$attr_key
mkdir -p $base_model_dir

now=`date +%Y%m%d_%H%M%S` 
model_dir=$base_model_dir/$now

#pretrain_model_dir=models/skirt_length_labels/baseline/
pretrain_model_dir=models/pretrain_resnet50/
pretrain_warm_vars='^((?!dense).)*$'
if [ "$attr_key" = "skirt_length_labels" ]; then
	pretrain_warm_vars='.*'
	pretrain_model_dir=model_skirt_pretrain/
fi

mkdir log

if [ "$mode" = "predict" ]; then
	train_or_predict="--predict"
fi

echo "mode: $mode"
echo "attr_key: $attr_key"
echo "model_dir: $model_dir"
echo "pretrain_model_dir: $pretrain_model_dir"
echo "pretrain_warm_vars: $pretrain_warm_vars"

nohup python app.py \
	--data_dir ~/data/vision/fashionAI/ \
	--model_dir $model_dir \
	--attr_key $attr_key \
	--pretrain_model_dir $pretrain_model_dir \
	--pretrain_warm_vars $pretrain_warm_vars \
	$train_or_predict \
	>log/${attr_key}_${now}.out 2>&1 &
