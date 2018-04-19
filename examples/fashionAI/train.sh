
attr_key=${1:-skirt_length_labels}
data_dir=${2:-~/data/vision/fashionAI/}
pretrain_model_dir=${3:-models/pretrain_resnet50/}
model_dir=${4:-models/$attr_key/baseline}

now=`date +%Y%m%d_%H%M` 

pretrain_warm_vars='^((?!dense).)*$'

log_dir=log/$attr_key
mkdir -p $log_dir

echo "attr_key: $attr_key"
echo "data_dir: $data_dir"
echo "model_dir: $model_dir"
echo "pretrain_model_dir: $pretrain_model_dir"
echo "pretrain_warm_vars: $pretrain_warm_vars"

nohup python app.py \
	--data_dir $data_dir \
	--model_dir $model_dir \
	--attr_key $attr_key \
	--pretrain_model_dir $pretrain_model_dir \
	--pretrain_warm_vars $pretrain_warm_vars \
	>$log_dir/train_${now}.out 2>&1 &
