+ use python instead of ipython(core dumped very often, unkown reason)

### train command
In the [Zhihu post](https://zhuanlan.zhihu.com/p/24767059), the train command:

```bash
python main.py --image_size 96 --output_size 48 --dataset anime --is_crop True --is_train True --epoch 300 --input_fname_pattern "*.jpg"
```

should be modified to:

```bash
python main.py --input_height 96 --output_height 48 --dataset anime --is_crop True --is_train True --epoch 300 --input_fname_pattern "*.jpg"
```

use the trained model

```bash
python main.py --input_height 96 --output_height 48 --dataset anime --is_crop True --is_train False --epoch 300 --input_fname_pattern "*.jpg"
```

because of the update on github code.
