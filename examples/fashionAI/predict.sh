
attr_key=${1:-skirt_length_labels}
model_dir=$2

now=`date +%Y%m%d` 
#output_file=output/$now/${attr_key}.csv
#output_dir=`dirname $output_file`
output_dir=output/$now/${attr_key}
mkdir -p $output_dir

echo "attr_key: $attr_key"
echo "model_dir: $model_dir"
echo "output_dir: $output_dir"

python app.py \
	--data_dir ~/data/vision/fashionAI/ \
	--model_dir $model_dir \
	--attr_key $attr_key \
	--predict \
	--predict_output_dir=$output_dir
