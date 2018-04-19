
attr_key=${1:-skirt_length_labels}
data_dir=${2:-~/data/vision/fashionAI/}
model_dir=${3:-models/$attr_key/baseline}

day=`date +%Y%m%d` 
now=`date +%Y%m%d_%H%M` 

output_dir=output/${attr_key}/$day
mkdir -p $output_dir

log_dir=log/$attr_key
mkdir -p $log_dir

echo "data_dir: $data_dir"
echo "attr_key: $attr_key"
echo "model_dir: $model_dir"
echo "output_dir: $output_dir"

nohup python app.py \
	--data_dir $data_dir \
	--model_dir $model_dir \
	--attr_key $attr_key \
	--predict \
	--predict_output_dir=$output_dir \
	>$log_dir/predict_${now}.out 2>&1 &
