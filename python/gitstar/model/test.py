import pipeline_invoke


with open('../input/predict/test_request.json', 'rb') as fh:
  contents_binary = fh.read()

print(pipeline_invoke.invoke(contents_binary))
