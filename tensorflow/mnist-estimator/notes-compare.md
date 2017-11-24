```
cfregly:0 cfregly$ saved_model_cli show --dir . --all
/Users/cfregly/miniconda3/lib/python3.6/importlib/_bootstrap.py:219: RuntimeWarning: compiletime version 3.5 of module 'tensorflow.python.framework.fast_tensor_util' does not match runtime version 3.6
  return f(*args, **kwds)

MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['prediction']:
The given SavedModel SignatureDef contains the following input(s):
inputs['inputs'] tensor_info:
    dtype: DT_STRING
    shape: (-1)
    name: input_example_tensor:0
The given SavedModel SignatureDef contains the following output(s):
outputs['classes'] tensor_info:
    dtype: DT_STRING
    shape: (-1)
    name: Cast:0
outputs['scores'] tensor_info:
    dtype: DT_FLOAT
    shape: (-1, 10)
    name: softmax_tensor:0
Method name is: tensorflow/serving/classify

signature_def['serving_default']:
The given SavedModel SignatureDef contains the following input(s):
inputs['inputs'] tensor_info:
    dtype: DT_STRING
    shape: (-1)
    name: input_example_tensor:0
The given SavedModel SignatureDef contains the following output(s):
outputs['classes'] tensor_info:
    dtype: DT_STRING
    shape: (-1)
    name: Cast:0
outputs['scores'] tensor_info:
    dtype: DT_FLOAT
    shape: (-1, 10)
    name: softmax_tensor:0
Method name is: tensorflow/serving/classify
```
